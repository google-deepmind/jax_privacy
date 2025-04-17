# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for distributed noise generation in JAX.

This library provides utilities for generating correlated noise efficiently
in distributed environments. The main entry point is
`streaming_matrix_to_sharded_privatizer`, which accepts a streaming matrix and
generates a `GradientPrivatizer` object that can be used to privatize
gradient streams.

This file is based on the implementation described in [Scaling up the Banded
Matrix Factorization Mechanism for Differentially Private
ML](https://arxiv.org/abs/2405.15913). It is designed to produce general
PyTree-structured noise, and is carefully designed to handle edge cases
correctly when the size of the leaves in the pytree do are not evenly
divisible by the number of devices.

The correlated noise produced by this library will be sharded according to a
specified `out_sharding'.  Internally, the state required to generate this
noise will be sharded across all devices in the mesh.  Therefore, the efficiency
and scalability of noise generation is primarily determined by the per-device
memory usage, rather than the total memory, and is hence dependent on the number
of devices.

This library relies heavy on JAX's
[Distributed arrays and automatic parallelization](
https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
"""

import functools
import math
from typing import Any, TypeAlias

import jax
from jax.experimental import shard_alike
from jax.experimental import shard_map
from jax_privacy.dp_sgd import typing
from jax_privacy.dpftrl_mechanisms import streaming_matrix
from jax_privacy.stream_privatization import additive_privatizers
from jax_privacy.stream_privatization import gradient_privatizer
import numpy as np


# pylint: disable=invalid-name

PyTree: TypeAlias = Any


def _padded_size(x: jax.Array) -> int:
  """Array size, padded to the next multiple of jax.device_count()."""
  unpadded = x.size
  num_devices = jax.device_count()
  remainder = unpadded % num_devices
  return unpadded + num_devices - remainder if remainder != 0 else unpadded


def _flatten_pspec(p: jax.sharding.PartitionSpec) -> jax.sharding.PartitionSpec:
  """Flatten a PartitionSpec from a nD sharding to an "equivalent" 1D sharding.

  Args:
    p: A PartitionSpec defined over a nD array.

  Returns:
    A PartitionSpec defined over the same devices / mesh axes as p, but defined
    wrt a flattened version of the original array.
  """
  # E.g., (None, ('x', 'y'), None, 'z') --> ('x', 'y', 'z').
  # E.g., ('data', None, ('replica', 'mdl')) -> ('data', 'replica', 'mdl')
  result = []
  for item in p:
    if isinstance(item, tuple):
      result.extend(item)
    elif isinstance(item, str):
      result.append(item)
    elif item is None:
      continue
    else:
      raise ValueError(f'Unexpected item in PartitionSpec: {item}.')
  return jax.sharding.PartitionSpec(tuple(result))


def _tree_unzip(tree, treedef):
  """Given a tree of tuples, return a tuple of trees."""
  leaves = treedef.flatten_up_to(tree)
  return tuple(treedef.unflatten(x) for x in zip(*leaves))


def _sharding_agnostic_reshape_add(x: jax.Array, y: jax.Array):
  """Reshapes y[:x.size] into x.shape and adds to x."""
  result = x + y[: x.size].reshape(x.shape).astype(x.dtype)
  return shard_alike.shard_alike(result, x)[0]


def _reshape_add(
    x: jax.Array,
    y: jax.Array,
    out_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Reshapes y[:x.size] into x.shape and x.sharding and adds to x.

  The output of this function will respect the out_sharding.  Generally,
  the input x should also be sharded according to out_sharding.  Internally,
  this function uses JAX's shard_map primitive to perform a local reshape on
  each device, avoiding the need to transfer data between devices.

  Note that this function is not quite equivalent to
  x + y[:x.size].reshape(x.shape).  In particular, the precise mapping between
  the entries of y and it's reshaped version is not in general the same
  as jnp.reshape.  We do guarantee (and test) that each entry of y will be
  mapped to at most one entry of the reshaped version.  Since y are typically
  i.i.d. random bits, the precise mapping is not important for our purposes.

  Args:
    x: The first array.  Should be sharded according to out_sharding.
    y: The second array.  Should be 1D, and sharded across all devices in the
      out_sharding.mesh, and have size greater than or equal to x.size.
    out_sharding: The sharding of the returned array.  If not specified, then
      the reshape may result in unncessary communication between devices.

  Returns:
    x + reshape(y[:x.size], x.shape), with sharding equal to out_sharding.
  """
  if x.size > y.size:
    # x.size and y.size must be evenly divisible by the number of devices in
    # their respective shardings.  x will generally be sharded over a
    # subset of devices (and replicated across the rest), while y will be
    # fully sharded across all devices.  Padding is required in settings where
    # y.size is not evenly divisible by the number of devices.
    raise ValueError(
        f'x.size ({x.size}) must be less than or equal to y.size ({y.size}).'
    )

  per_device_shape = out_sharding.shard_shape(x.shape)
  per_device_size = math.prod(per_device_shape)

  z = shard_map.shard_map(
      lambda v: v[:per_device_size].reshape(per_device_shape),
      mesh=out_sharding.mesh,
      # Here the input will be replicated across the mesh axes not
      # in out_sharding.spec (as desired).
      in_specs=_flatten_pspec(out_sharding.spec),
      out_specs=out_sharding.spec,
  )(y)
  return jax.lax.with_sharding_constraint((x + z).astype(x.dtype), out_sharding)


def _infer_state_sharding(
    noising_matrix: streaming_matrix.StreamingMatrix,
    internal_flat_sharding: jax.sharding.NamedSharding,
) -> jax.sharding.NamedSharding:
  """Infer the sharding of the state for a given noising matrix.

  Assumptions:
    - The noising_matrix will be used to generate 1D noise slices.
    - The internal_flat_sharding specifies how these noise slices should be
    sharded.
    - The state of the streaming matrix includes some arrays which have an axis
    of size equal to the size of the 1D noise slices to be produced.  If this
    assumption is violated, then the state will be fully replicated.

  Args:
    noising_matrix: The noising matrix to use.
    internal_flat_sharding: The sharding to use for the 1D noise slices.

  Returns:
    A sharding specification for the state of the streaming matrix.
  """
  def sharding_for_dummy_array(x):
    pspecs = [None]*x.ndim
    for i in range(x.ndim):
      if x.shape[i] == 2147483647 and len(internal_flat_sharding.spec) == 1:
        pspecs[i] = internal_flat_sharding.spec[0]
        break
    return jax.sharding.NamedSharding(
        internal_flat_sharding.mesh, jax.sharding.PartitionSpec(*pspecs)
    )

  dummy_state = jax.eval_shape(
      functools.partial(noising_matrix.init_multiply, (2147483647,))
  )
  return jax.tree.map(sharding_for_dummy_array, dummy_state)


def streaming_matrix_to_sharded_privatizer(
    noising_matrix: streaming_matrix.StreamingMatrix,
    stddev: float,
    out_sharding: PyTree | None,
    internal_flat_shardings: PyTree | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> gradient_privatizer.GradientPrivatizer:
  """Construct a GradientPrivatizer from a streaming matrix.

  The returned GradientPrivatizer is internally designed to produce noise
  sharded according to the out_sharding, which should match the PyTree
  structure and shardings of the model and gradients.  However, the state
  required to produce this noise is sharded across all devices in the mesh,
  which allows this library to scale up to large noise states that may be
  encountered when running e.g., Unamplified BandMF.  In general, the
  performance characteristics of the returned GradientPrivatizer is determined
  primarily by the per-device memory usage of the internal state:

  ```
  GB_PER_FLOAT = 4 / 2**30  # Assuming float32 state
  PER_DEVICE_GB = MODEL_SIZE * NUM_BUFFERS * GB_PER_FLOAT / NUM_DEVCIES
  ```

  Here, MODEL_SIZE is the number of model parameters, NUM_BUFFERS is the number
  of buffers in the streaming matrix (number of bands for BandMF), and
  NUM_DEVICES is the number of devices in the mesh.

  NOTE: When using this function in a distributed environment, please ensure to
  enable a "partitionable" random number generator, e.g. via the
  'jax_threefry_partitionable' config option.  For more information, see
  https://jax.readthedocs.io/en/latest/jax.random.html#advanced-rng-configuration.
  See jax_privacy/examples/distributed_noise_generation.py for an example
  of how to use this function and visualize the array shardings.

  Args:
    noising_matrix: A streaming matrix representing the matrix $C^{-1}$.
    stddev: The standard deviation of the (uncorrelated) source noise. That is,
      the returned GradientPrivatizer will add noise corresponding to the rows
      of `noising_matrix @ Z`, where Z is a matrix of iid Gaussian noise with
      standard deviation `stddev`.
    out_sharding: A PyTree of NamedShardings for the model params. The PyTree
      structure of the out_sharding must match the structure of model. The
      correlated noise added by the returned GradientPrivatizer. will be
      generated according to these shardings as well. Internally this noise will
      generally be sharded in a different manner before it is resharded to the
      out_sharding. While out_sharding=None is a valid argument, it is not
      recommended in distributed settings since it will result in a reshape
      operation that may lead to unncessary communication. If the number of
      trainable parameters is small (e.g., with LoRA finetuning), then this
      overhead is probably acceptable.
    internal_flat_shardings: A PyTree of NamedShardings to use internally to
      shard the flattened iid noise vectors and flattened correlated noise
      vectors. If None, then the internal flat shardings will be determined
      automatically and utilize all mesh axes. This argument also determines how
      the correlated noise state is sharded. Specifically, each row of the noise
      state will be sharded according to internal_flat_shardings.
    mesh: A Mesh to use to define the internal_flat_shardings.

  Returns:
    A GradientPrivatizer object.
  """
  is_named_sharding = lambda l: isinstance(l, jax.sharding.NamedSharding)
  if not jax.tree.all(jax.tree.map(is_named_sharding, out_sharding)):
    raise ValueError(
        f'out_sharding must be a PyTree of NamedShardings.  Got {out_sharding}.'
    )
  if out_sharding is None and internal_flat_shardings is None and mesh is None:
    raise ValueError(
        'mesh must be specified if neither internal_flat_shardings nor'
        ' out_sharding are given.'
    )

  # We will generate fully sharded noise in 1 dimension, and then transform
  # it to the correct shape and sharding after applying the noising matrix.
  # It might also be natural to shard the independent noise in the same manner
  # as the out_sharding.  The main downside of this approach is that
  # after generation, the pytree of independent noise would need to be
  # immediately flattened, possibly padded, and resharded in order to
  # incorporate into the state.  Thus, the shards of the array would potentially
  # need to be moved between devices, which could be inefficient in some cases.
  if internal_flat_shardings is None:
    if mesh is not None:
      internal_flat_shardings = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(mesh.axis_names)
      )
    else:
      internal_flat_shardings = jax.tree.map(
          lambda s: jax.sharding.NamedSharding(
              # 1D array is fully sharded across all mesh axes.
              s.mesh,
              jax.sharding.PartitionSpec(s.mesh.axis_names),
          ),
          out_sharding,
      )

  def init(mdl_params: typing.ParamsT) -> typing.NoiseStateT:
    # We determine the size of x and round up to the nearest multiple of
    # jax.device_count(), which will enable us to fully shard the columns
    # of the state.  The padded correlated noise values produced can safely
    # be discarded since it is independent across index (even though it is
    # correlated across iterations).

    def init_leaf(x, shd):
      size = (_padded_size(x),)
      # Converting to array is needed when leaves are scalar primitives.
      state = jax.tree.map(jax.numpy.array, noising_matrix.init_multiply(size))
      sharding = _infer_state_sharding(noising_matrix, shd)
      return jax.lax.with_sharding_constraint(state, sharding)

    if isinstance(internal_flat_shardings, jax.sharding.NamedSharding):
      shardings = jax.tree.map(lambda _: internal_flat_shardings, mdl_params)
    else:
      shardings = internal_flat_shardings

    return jax.tree.map(init_leaf, mdl_params, shardings)

  def privatize(
      *,
      sum_of_clipped_grads: typing.ParamsT,
      noise_state: typing.NoiseStateT,
      prng_key: jax.Array,
  ) -> tuple[typing.ParamsT, typing.NoiseStateT]:
    tree_state = noise_state

    noise_shapes = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct((_padded_size(x),), x.dtype),
        sum_of_clipped_grads,
    )

    iid_noise = jax.lax.with_sharding_constraint(
        additive_privatizers.isotropic_gaussian_noise_fn(
            rng=prng_key,
            index=0,
            stddev=stddev,
            example_grad=noise_shapes,
        ),
        internal_flat_shardings,
    )

    correlated_noise, state = _tree_unzip(
        jax.tree.map(noising_matrix.multiply_next, iid_noise, tree_state),
        jax.tree.structure(iid_noise),
    )

    if out_sharding is None:
      noisy_grad = jax.tree.map(
          _sharding_agnostic_reshape_add, sum_of_clipped_grads, correlated_noise
      )
    else:
      noisy_grad = jax.tree.map(
          _reshape_add, sum_of_clipped_grads, correlated_noise, out_sharding
      )

    return noisy_grad, state

  return gradient_privatizer.GradientPrivatizer(init, privatize)


def streaming_matrix_to_single_machine_privatizer(
    noising_matrix: streaming_matrix.StreamingMatrix,
    stddev: float,
) -> gradient_privatizer.GradientPrivatizer:
  """Construct a GradientPrivatizer from a streaming matrix.

  Args:
    noising_matrix: A streaming matrix representing the matrix $C^{-1}$.
    stddev: The standard deviation of the (uncorrelated) source noise. That is,
      the returned GradientPrivatizer will add noise corresponding to the rows
      of `noising_matrix @ Z`, where Z is a matrix of iid Gaussian noise with
      standard deviation `stddev`.

  Returns:
    A GradientPrivatizer object.
  """
  if jax.device_count() != 1:
    raise ValueError(
        'This function is only intended to be used on a single machine.'
        'Use streaming_matrix_to_sharded_privatizer in distributed settings.'
    )
  return streaming_matrix_to_sharded_privatizer(
      noising_matrix,
      stddev,
      out_sharding=None,
      mesh=jax.sharding.Mesh(np.array(jax.devices()[0]), ()),
  )
