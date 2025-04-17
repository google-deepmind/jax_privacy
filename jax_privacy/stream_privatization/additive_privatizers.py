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

"""Implementations of GradientPrivatizer constructors which add noise."""

from collections.abc import Callable

import chex
import jax
from jax import numpy as jnp
from jax_privacy.stream_privatization import gradient_privatizer
import jaxtyping


IntScalar = jaxtyping.Int[jaxtyping.Array, '']
FloatScalar = jaxtyping.Float[jaxtyping.Array, '']


def isotropic_gaussian_noise_fn(
    rng: jax.Array,
    index: IntScalar,
    stddev: FloatScalar,
    example_grad: chex.ArrayTree,
) -> chex.ArrayTree:
  """Samples from isotropic Gaussian noise via the key (rng, index).

  This function guarantees that unique (rng, index) pairs return fresh
  pseudorandom samples from an isotropic Gaussian, and that the same (rng,
  index) pairs return the same values.

  When generating correlated noise for DP-FTRL, this function is intended to
  generate the slices of the (uncorrelated) right-hand-side noise Z before
  multiplying by the noise_correlating_matrix.

  Args:
    rng: Instance of `jax.random.PNRGKeyArray`, used to seed the noise
      generation process.
    index: Integer specifying the index of the slice to be computed, e.g., the
      row in the matrix C^{-1}. Used as a combined key with `rng`.
    stddev: Float specifying the standard deviation of the isotropic Gaussian
      from which to sample.
    example_grad: Nested arrays determining the shape of the sample to be
      computed. The arrays themselves may be either concrete or abstract, e.g.,
      instances of `jax.ShapeDtypeStruct` rather than literal arrays.

  Returns:
    A sample from an isotropic Gaussian matching the structure of
    `example_grad`, as described above.
  """
  flat_examples, examples_treedef = jax.tree_util.tree_flatten(example_grad)
  combined_key = jax.random.fold_in(rng, index)
  rngs = jax.random.split(combined_key, len(flat_examples))
  noise = [
      stddev * jax.random.normal(r, arr.shape, arr.dtype)
      for r, arr in zip(rngs, flat_examples)
  ]
  return jax.tree_util.tree_unflatten(examples_treedef, noise)


def _index_state_init(params: chex.ArrayTree) -> IntScalar:
  # We don't need params here vecause we only track index, re-materializing
  del params  # Unused.
  return jnp.array(0)


def gaussian_privatizer(
    *,
    stddev: FloatScalar,
    example_grad: chex.ArrayTree,
) -> gradient_privatizer.GradientPrivatizer:
  """Constructs `GradientPrivatizer` adding isotropic Gaussian noise.

  The Gaussian samples this privatizer adds to its inputs will be doubly keyed,
  by noise_key and the state of the privatizer (which in this case is just an
  index).

  Args:
    stddev: Standard deviation of the Gaussians to add.
    example_grad: Example arrays (concrete or abstract) defining the shape and
      dtype of Gaussian noise to generate.

  Returns:
    A `GradientPrivatizer` yielding samples of isotropic Gaussian noise,
    of shape and dtype corresponding to `example_grad`.
  """

  def privatize(
      sum_of_clipped_grads: chex.ArrayTree,
      noise_state: chex.ArrayTree,
      prng_key: jax.Array,
  ) -> tuple[chex.ArrayTree, IntScalar]:
    noise = isotropic_gaussian_noise_fn(
        rng=prng_key, index=0, stddev=stddev, example_grad=example_grad
    )
    privatized_grads = jax.tree.map(jnp.add, sum_of_clipped_grads, noise)
    return privatized_grads, noise_state

  return gradient_privatizer.GradientPrivatizer(lambda _: (), privatize)


# TODO: b/375481399 - Unify these matrix-check and related functions throughout
# jax_privacy.
def _check_matrix(array: jnp.ndarray):
  if array.ndim != 2:
    raise ValueError(
        'Expected matrix (a jnp.ndarray with length-2 shape); '
        'found an array whose shape is of length '
        f'{len(array.shape)}'
    )


def _compute_loop_bounds(matrix_row: chex.Array) -> tuple[IntScalar, IntScalar]:
  """Computes bounds for jax.lax.fori_loop version of row-column reduction.

  This function computes the indices to slice a vector to its largest span of
  non-zero entries. This is particularly useful for banded matrices where we
  have a smaller non-zero vector contained within a larger vector with zero
  entries at its edges. In doing so, we can reduce wasteful vector-matrix
  products when they would otherwise result in a zero entry anyway.

  Args:
    matrix_row: Vector array representing the matrix row by which to multiply.

  Returns:
    A pair of integer-scalar arrays, representing minimal and maximal indices
    over which we need to reduce in loop-based matmul computation, as described
    above.
  """
  if matrix_row.ndim != 1:
    raise ValueError(
        'Matrix row expected to be a vector, got tensor of '
        f'rank {len(matrix_row.shape)} and shape {matrix_row.shape}.'
    )
  nonzero_entries = (matrix_row != 0).astype(jnp.int32)
  first_nonzero = jnp.argmax(nonzero_entries)
  # We reverse to get the last index.
  last_nonzero = matrix_row.shape[0] - jnp.argmax(nonzero_entries[::-1])
  return first_nonzero, last_nonzero


def _multiply_matrix_row_with_noise_generator(
    matrix_row: jnp.ndarray,
    noise_generator: Callable[[int | IntScalar, chex.Array], chex.Array],
    result_shapes: chex.Array,
) -> chex.Array:
  """Matrix row multiply with a generator yielding row-slices of a tensor.

  Implements pseudocode:

  ```python
  accumulator = 0.0
  for i in range(first_nonzero(matrix_row), last_nonzero(matrix_row) + 1):
    accumulator += matrix_row[i] * noise_generator(i)
  return accumulator
  ```

  Args:
    matrix_row: Rank-1 array representing the row of a matrix by which we are
      multiplying.
    noise_generator: Function accepting integers (or scalar arrays of dtype int)
      along with a ShapeDtypeStruct, returning slices over the zeroth dimension
      of the tensor we are multiplying (IE, the contraction dimension).
    result_shapes: The shapes of the result to accumulate into.

  Returns:
    The result of performing the row-tensor multiplication as specified above.
  """

  def loop_body(idx, reduction_state):
    partial = reduction_state
    row_element = matrix_row[idx]
    next_slice = noise_generator(idx, result_shapes)

    def _return_noised_partial(noise_slice, partial_sum):
      return jax.tree.map(
          lambda x, y: jnp.astype(row_element * x + y, y.dtype),
          noise_slice,
          partial_sum,
      )

    def _return_partial(noise_slice, partial_sum):
      del noise_slice  # Unused
      return partial_sum

    # If the row_element is zero, skip the multiply-add.
    return jax.lax.cond(
        row_element != 0,
        _return_noised_partial,
        _return_partial,
        next_slice,
        partial,
    )

  # Since we make the bounds of the loop dynamic (dependent on the value of
  # `matrix_row`), the loop here will not work under jax.grad (or jax.vjp).
  lower_bound, upper_bound = _compute_loop_bounds(matrix_row)
  loop_state = jax.tree.map(jnp.zeros_like, result_shapes)
  reduced = jax.lax.fori_loop(lower_bound, upper_bound, loop_body, loop_state)
  return reduced


def concrete_correlated_gaussian_privatizer(
    *,
    noise_correlation_matrix: jnp.ndarray,
    noise_key: jax.Array,
    stddev: float,
) -> gradient_privatizer.GradientPrivatizer:
  """Creates a `GradientPrivatizer` adding DP-MF-FTRL noise to gradients.

  Performs on-the-fly noise generation by iterating the non-zero rows of the
  noise_correlation_matrix. This is only practical when the scale is relatively
  small (<10,000 columns). This is not reverse-mode autodiff compatible, but
  may take advantage of banded structure in `noise_correlation_matrix`. It is
  primarily intended for simple experimentation, where e.g. scale of
  `noise_correlation_matrix` is not too large, or as a reference for testing.

  For naming of these parameters, see ../dpftrl_mechanisms/README.md.

  Args:
    noise_correlation_matrix: Rank-2 `ndarray` (IE, a noise_correlation_matrix)
      with which to multiply the right-hand side argument.
    noise_key: PRNGKey array representing the key to use for the
      noise-generation of this privatizer.
    stddev: Standard deviation to use for the noise of this privatizer.

  Returns:
    A `GradientPrivatizer` which adds samples from Gaussian correlated by
    `noise_correlation_matrix` (IE, samples from a Gaussian with covariance
    `noise_correlation_matrix.T @ noise_correlation_matrix`),
    keyed by `noise_key`, to its stream of gradients.
  """

  _check_matrix(noise_correlation_matrix)

  correlated_noise_fn = jax.jit(
      _multiply_matrix_row_with_noise_generator,
      static_argnums=1,
  )

  def _gaussian_for_index(idx, example_grad):
    return isotropic_gaussian_noise_fn(
        rng=noise_key, index=idx, stddev=stddev, example_grad=example_grad
    )

  def privatize(
      sum_of_clipped_grads: chex.ArrayTree,
      noise_state: IntScalar,
      prng_key: jax.Array,
  ) -> tuple[chex.ArrayTree, IntScalar]:
    # TODO: b/394128304 - refactor this.
    del prng_key  # Unused.
    index = noise_state
    noise_correlation_matrix_row = noise_correlation_matrix[index, :]
    noise = correlated_noise_fn(  # pylint: disable=not-callable
        noise_correlation_matrix_row, _gaussian_for_index, sum_of_clipped_grads
    )
    privatized_grads = jax.tree.map(jnp.add, noise, sum_of_clipped_grads)
    return privatized_grads, index + 1

  return gradient_privatizer.GradientPrivatizer(_index_state_init, privatize)
