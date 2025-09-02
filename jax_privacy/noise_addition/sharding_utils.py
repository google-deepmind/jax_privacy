# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
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

"""Utilities for sharding in multi-machine settings.

This file houses helper functions related to sharding. Users of JAX Privacy
should not need to use this file directly, but the utilites implemented here
are leveragedd by higher-level APIs elsewhere in the library. The functions
defined here assume that input arrays are enriched with type-level sharding
information, as described in
https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html.

This file is contains primitives needed for "distributed noise generation"
as described in [Scaling up the Banded Matrix Factorization Mechanism for
Differentially Private ML](https://arxiv.org/abs/2405.15913) and in
[Correlated Noise Mechanisms for Differentially Private Learning]
(https://arxiv.org/abs/2506.08201).
"""

import math
from typing import Any, TypeAlias
import jax

PyTree: TypeAlias = Any
PartitionSpecPyTree: TypeAlias = Any


def _ceiling_to_multiple(size: int, multiple: int) -> int:
  """Return the smallest multiple of multiple that is >= size."""
  remainder = size % multiple
  return size + multiple - remainder if remainder != 0 else size


def flatten_with_zero_redundancy(abstract_array: jax.Array) -> jax.Array:
  """Return a flattened, padded, and ZeRo-sharded abstract version of x.

  Specifically, the returned object will describe a 1D array that is
  partitioned across all mesh axes. This is precisely the concept of
  "zero-redundancy" sharding as described in https://arxiv.org/pdf/1910.02054,
  i.e., no redundant copies of the partitioned value exist anywhere.

  Args:
    abstract_array: Abstract (or concrete) array with shape, dtype, and sharding
      attributes.

  Returns:
    A zero-redundancy abstract flattened+padded version of the input value.
  """
  mesh = jax.typeof(abstract_array).sharding.mesh
  # As of JAX 0.7.0, jnp.*_like will not preserve sharding of ShapeDtypeStruct
  # defined w.r.t. AbstractMeshes, so we return a concrete array here.
  # Under JIT, this should get optimized away.
  # TODO: b/415360727 - Version bump to 0.7.1, swap in jax.ShapeDtypeStruct,
  # and add type annotations to this function.
  return jax.numpy.empty(
      _ceiling_to_multiple(abstract_array.size, mesh.size),
      dtype=abstract_array.dtype,
      out_sharding=jax.sharding.NamedSharding(mesh, jax.P(mesh.axis_names)),
  )


def _flatten_pspec(p: jax.sharding.PartitionSpec) -> jax.sharding.PartitionSpec:
  """Flatten a PartitionSpec from a nD sharding to an "equivalent" 1D sharding.

  Example Usage:
    >>> p = jax.sharding.PartitionSpec(None, ('x', 'y'), None, 'z')
    >>> _flatten_pspec(p)
    PartitionSpec(('x', 'y', 'z'),)

  Example Usage:
    >>> p = jax.sharding.PartitionSpec('data', None, ('replica', 'mdl'))
    >>> _flatten_pspec(p)
    PartitionSpec(('data', 'replica', 'mdl'),)

  Args:
    p: A PartitionSpec defined over a nD array.

  Returns:
    A PartitionSpec defined over the same devices / mesh axes as p, but defined
    wrt a flattened version of the original array.
  """
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


def local_reshape_add(x: jax.Array, y: jax.Array) -> jax.Array:
  """Reshapes y[:x.size] into x.shape and x.sharding and adds to x.

  This function expects both x and y to have type-level sharding information,
  See https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html for more
  details.

  Internally, this function uses JAX's shard_map primitive to perform a local
  reshape on each device, avoiding the need to transfer data between devices
  that would arise if using jnp.reshape.

  Note that this function is not quite equivalent to
  x + y[:x.size].reshape(x.shape).  In particular, the precise mapping between
  the entries of y and it's reshaped version is not in general the same
  as jnp.reshape.  We do guarantee (and test) that each entry of y will be
  mapped to at most one entry of the reshaped version.  Since y are typically
  i.i.d. random bits, the precise mapping is not important for our purposes.

  Formal Guarantees:
    - For each index tuple i and j, z_i = x_i + y_j for some j. y_j is only
      mapped to at most one entry of z.
    - z has the same shape, dtype, and sharding as x.

  Args:
    x: The first array.
    y: The second array.  Should be 1D, and sharded across all devices in the
      out_sharding.mesh, and have size greater than or equal to x.size.

  Returns:
    x + reshape(y[:x.size], x.shape), with sharding equal to out_sharding.
  """
  out_sharding = jax.typeof(x).sharding
  per_device_shape = out_sharding.shard_shape(x.shape)
  per_device_size = math.prod(per_device_shape)

  reshape = jax.shard_map(
      lambda v: v[:per_device_size].reshape(per_device_shape),
      mesh=out_sharding.mesh,
      # Replicates input across mesh axes not in out_sharding.spec (as desired).
      in_specs=_flatten_pspec(out_sharding.spec),
      out_specs=out_sharding.spec,
  )
  return (x + reshape(y)).astype(x.dtype)


