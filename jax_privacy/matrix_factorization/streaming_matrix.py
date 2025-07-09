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

"""Definition of streamin matrix interface."""

from collections.abc import Callable
from typing import TypeVar
import jax
from jax import numpy as jnp
import jaxtyping


MatrixArray = jaxtyping.Num[jaxtyping.Array, 'dim1 dim2']
State = TypeVar('State')

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


class StreamingMatrix:
  """A linear mapping x -> A x for a lower-triangular (streaming) A matrix.

  Via the member functions `init_multiply` and `next_multiply_element`,
  this class allows you to efficiently compute a linear mapping x -> A x
  in streaming fashion (one element at a time). The precise meaning of the term
  `efficienctly` is implementation-dependent, with examples including constant
  memory overhead, and / or without fully materializing A or x.

  Example:
  >>> A = prefix_sum()
  >>> x = jnp.arange(1, 5).astype(float)
  >>> slices = []
  >>> state = A.init_multiply(x[0].shape)
  >>> for i in range(len(x)):
  ...   result_slice, state = A.multiply_next(x[i], state)
  ...   slices.append(result_slice)
  >>> Ax = jnp.stack(slices)
  >>> print(Ax)
  [ 1.  3.  6. 10.]
  >>> print(jnp.cumsum(x))
  [ 1.  3.  6. 10.]

  See the constructor docstring for a full description of `init_multiply` and
  `next_multiply_element`.

  Importantly, this design encodes the fact that Ax[i] may only depend on
  x[i] and state captured from computing Ax[0], ..., Ax[i-1]. This is equivalent
  to `A` having a lower-triangular matrix representation in the standard basis.

  In general, `A` and `x` may both be infinite; thus we sidestep the
  question of how many elements of `Ax` one wishes to compute by assuming the
  user provides a range.
  """

  def __init__(
      self,
      init_multiply_fn: Callable[[tuple[int, ...]], State],
      multiply_next_fn: Callable[[jax.Array, State], tuple[jax.Array, State]],
  ):
    """Construct a StreamingMatrix object.

    Naming note: In general, the `rhs_row_shape` need not be a vector, but could
    be an arbitrary rank tensor. For simplicity we use the name `row`
    for `x[i]` and `y[i]` as this captures the most common case, and via
    suitable flattening one can without loss of generality take the RHS
    `x` to be a matrix.

    Design note: We take the init_fn and next_fn as arguments rather than
    defining these as abstract methods in order to make it easy to
    generate different `StreamingMatrix` implementations
    "on the fly" without subclassing. This class is not intended to be
    subclassed.

    Args:
      init_multiply_fn: A function that returns the initial state given the
        expected shape of inputs to each call to next_fn. For example, if
        multiplying this `LowerTriangularMatrix` with an n x n matrix X, the
        shape would be (n,) corresponding to each row of X. This function should
        be jax-jittable.
      multiply_next_fn: A function that returns (next_slice, updated_state) from
        (next_input, current_state). This function should be jax-jittable.
    """
    self.init_multiply = init_multiply_fn
    self.multiply_next = multiply_next_fn

  def materialize(self, n: int) -> MatrixArray:
    """A utility method to materialize this matrix as an n x n ndarray.

    Note `n` needs to be a parameter, because a general `StreamingMatrix`
    can represent an infinite-dimensional matrix.

    NOTE: Primarily for debugging and testing implementations of init and next.

    Args:
      n: The size of the square matrix to materialize.

    Returns:
      An n x n materialization of this matrix.
    """
    return self @ jnp.eye(n)

  def row_norms_squared(self, n: int, scan_fn=jax.lax.scan) -> jax.Array:
    """Computes the row-wise L2^2 norm of the matrix.

    Given a StreamingMatrix B = A C^{-1}, this function computes the per-query
    expected squared error of the factorization A = BC.  The expected total
    squared error and the maximum expected squared error can be computed from
    this vector via jnp.sum and jnp.max respectively.

    This function consumes an optional scan_fn argument, which is primarily
    useful if you need to backpropagate through this function, in which case
    using a checkpointed scan can be helpful to avoid OOMing on GPUs.
    For example, the scan_fn defined below would store 8 intermediate states
    of the scan in memory, rather than the default which stores the entire
    scan history in memory during backpropagation.  The number of checkpoints
    determines the computation / memory tradeoff.

    ```
    scan_fn = functools.partial(
      equinox.internal.scan,
      kind='checkpointed',
      checkpoints=8,
    )
    ```

    Args:
      n: The number of rows to compute squared-norms of.
      scan_fn: A function with the same signature as jax.lax.scan.

    Returns:
      A vector of length n containing the row-wise L2^2 norm of the matrix.
    """
    zero = jnp.zeros(n)

    def next_state_and_row_norm(state, i):
      # Note: state is first for use in scan_fn.
      ei = zero.at[i].set(1)
      row, state = self.multiply_next(ei, state)
      return state, row @ row

    return scan_fn(
        next_state_and_row_norm, self.init_multiply((n,)), jnp.arange(n)
    )[1]


# TODO: b/329444015 - Consider making protected and updating callsites
# to call the member-function directly.
def scale_rows_and_columns(
    matrix: StreamingMatrix,
    row_scale: jax.Array | None = None,
    col_scale: jax.Array | None = None,
) -> StreamingMatrix:
  """Returns a new `StreamingMatrix` with scaled rows and/or cols.

  Assumes row_scale and col_scale can be indexed into for as many outputs
  are generated from matrix. If `jax.Array`s are used, note
  row_scale[i] for i > len(row_scale) will return row_scale[-1].

  Args:
    matrix: The matrix to wrap.
    row_scale: Multipliers to apply to the rows of `matrix`, equivalent to
      jnp.diag(row_scale) @ matrix.
    col_scale: Multipliers to apply to the columns of `matrix, equivalent to
      matrix @ jnp.diag(col_scale).

  Returns:
    The wrapped `StreamingMatrix`.
  """

  def init(shape: tuple[int, ...]):
    return 0, matrix.init_multiply(shape)

  def next_fn(xi: jax.Array, state: State) -> tuple[jax.Array, State]:
    i, inner_state = state
    if col_scale is not None:
      xi = xi * col_scale[i]
    result, inner_state = matrix.multiply_next(xi, inner_state)
    if row_scale is not None:
      result = result * row_scale[i]
    return result, (i + 1, inner_state)

  return StreamingMatrix(init, next_fn)


def multiply_array(A: StreamingMatrix, x: jax.Array) -> jax.Array:
  """Computes the matrix-vector product A x."""

  # Reverse (value, state) -> (state, value) for scan.
  def f(state, value):
    return A.multiply_next(value, state)[::-1]

  return jax.lax.scan(f, A.init_multiply(x.shape[1:]), x)[1]


# TODO: b/329444015 - Consider making protected and updating callsites
# to call the member-function directly.
def multiply_streaming_matrices(
    A: StreamingMatrix,
    B: StreamingMatrix,
) -> StreamingMatrix:
  """Multiply a StreamingMatrix by another StreamingMatrix.

  Args:
    A: The left hand side matrix
    B: The right hand side matrix

  Returns:
    A B, represented as another StreamingMatrix.
  """

  def init_multiply(shape=()):
    return A.init_multiply(shape), B.init_multiply(shape)

  def multiply_next(value, state):
    A_state, B_state = state
    inner, B_state = B.multiply_next(value, B_state)
    outer, A_state = A.multiply_next(inner, A_state)
    return outer, (A_state, B_state)

  return StreamingMatrix(init_multiply, multiply_next)


def identity() -> StreamingMatrix:
  """An implicit representation of the identity matrix."""
  return StreamingMatrix(lambda _: (), lambda value, _: (value, ()))


def prefix_sum() -> StreamingMatrix:
  """An implicit representation of the lower triangular matrix of ones."""

  def init_multiply(shape: tuple[int, ...]) -> jax.Array:
    return jnp.zeros(shape)

  def multiply_next(
      state: jax.Array, value: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    result = state + value
    return result, result

  return StreamingMatrix(init_multiply, multiply_next)


def diagonal(diag: jax.Array) -> StreamingMatrix:
  """An implicit representation of a diagonal matrix.

  The returned StreamingMatrix represents an infinitely large diagonal matrix.
  The diagonal elements are taken from the provided array `diag` up to row
  n = diag.size, and is equal to diag[-1] beyond that point.

  Args:
    diag: A 1D array of diagonal elements.

  Returns:
    A StreamingMatrix representing the corresponding diagonal matrix.
  """
  return StreamingMatrix(
      lambda _: 0, lambda value, i: (value * diag.at[i].get(mode='clip'), i + 1)
  )


def momentum_sgd_matrix(
    momentum: float = 0, learning_rates: jax.Array | None = None
) -> StreamingMatrix:
  """An implicit representation of the momentum sgd matrix."""
  lr_sched = jnp.ones(1) if learning_rates is None else learning_rates
  if lr_sched.min() <= 0.0:
    raise ValueError(
        'Learning rates must be positive (zero learning rates may prevent '
        f'matrix factorization from succeeding.) Found {learning_rates}'
    )

  def init_multiply(shape: tuple[int, ...]) -> tuple[int, jax.Array, jax.Array]:
    return 0, jnp.zeros(shape), jnp.zeros(shape)

  def multiply_next(
      value: jax.Array, state: tuple[int, jax.Array, jax.Array]
  ) -> tuple[jax.Array, tuple[int, jax.Array, jax.Array]]:
    index, momentum_buf, result = state
    momentum_buf = momentum * momentum_buf + value
    # If index is out-of-bounds, return the last value in the array.
    result = result + lr_sched.at[index].get(mode='clip') * momentum_buf
    updated_state = (index + 1, momentum_buf, result)
    return result, updated_state

  return StreamingMatrix(init_multiply, multiply_next)


T = TypeVar('T', StreamingMatrix, jax.Array)


def _multiply_any(self: StreamingMatrix, other: T) -> T:
  """Multiply a StreamingMatrix by an array or a StreamingMatrix."""
  if isinstance(other, StreamingMatrix):
    return multiply_streaming_matrices(self, other)
  elif isinstance(other, jax.Array):
    return multiply_array(self, other)
  else:
    raise ValueError(f'Unsupported type for multiplication: {type(other)}')


def _multiply_scalar(self: StreamingMatrix, other: float) -> StreamingMatrix:
  """Multiply a StreamingMatrix by a scalar."""
  return multiply_streaming_matrices(self, diagonal(jnp.array([other])))


# Add some syntax sugar.
StreamingMatrix.__matmul__ = _multiply_any
StreamingMatrix.__mul__ = _multiply_scalar
StreamingMatrix.scale_rows_and_columns = scale_rows_and_columns
