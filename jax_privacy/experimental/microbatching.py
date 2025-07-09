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

"""A module for applying a function in a microbatched manner.

See README.md for more details.
"""
import dataclasses
import enum
import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


class AccumulationType(enum.Enum):
  SUM = enum.auto()
  MEAN = enum.auto()
  CONCAT = enum.auto()


def _astype(pytree: Any, dtype: jax.typing.DTypeLike | None = None) -> Any:
  if dtype is None:
    return pytree
  return jax.tree.map(lambda x: x.astype(dtype), pytree)


@dataclasses.dataclass
class Accumulator:
  """A class for accumulating values in a microbatched function."""

  num_microbatches: int
  accumulation_types: Any
  dtype: jax.typing.DTypeLike | None = None

  def init(self, value: Any):
    """Initialize the carry from an initial value."""
    return jax.tree.map(self._init, self.accumulation_types, value)

  def update(self, carry: Any, value: Any, index: int):
    """Update the carry with the new value."""
    update_fn = functools.partial(self._update, index=index)
    return jax.tree.map(update_fn, self.accumulation_types, carry, value)

  def finalize(self, carry: Any):
    """Process the final carry to get the correctly accumulated result."""
    return jax.tree.map(self._finalize, self.accumulation_types, carry)

  def _init(self, kind: AccumulationType, value: Any):
    value = _astype(value, self.dtype)
    match kind:
      case AccumulationType.SUM | AccumulationType.MEAN:
        return value
      case AccumulationType.CONCAT:
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num_microbatches,) + x.shape),
            value
        )

  def _update(self, kind: AccumulationType, state: Any, value: Any, index: int):
    match kind:
      case AccumulationType.SUM | AccumulationType.MEAN:
        return jax.tree.map(jnp.add, state, value)
      case AccumulationType.CONCAT:
        return jax.tree.map(
            lambda old, new: old.at[index].set(new), state, value
        )

  def _finalize(self, kind: AccumulationType, value: Any):
    match kind:
      case AccumulationType.SUM:
        return value
      case AccumulationType.MEAN:
        return jax.tree.map(lambda x: x / self.num_microbatches, value)
      case AccumulationType.CONCAT:
        return jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:], order='F'), value
        )


def _sharding_aware_reshape(pytree: Any, microbatch_size: int):
  """Reshape pytree leaves to shape (num_microbatches, microbatch_size, ...)."""
  # If data is sharded along the 0th axis, using column-major order is important
  # to ensure that each microbatch is sharded in the same manner.
  # For example, if the data was sharded across 2 devices, each device would
  # handle one of the examples in each microbatch.
  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] --> [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
  return jax.tree.map(
      lambda x: x.reshape(-1, microbatch_size, *x.shape[1:], order='F'),
      pytree,
  )


def _calculate_num_real_microbatches(
    is_padding_example: jax.Array,
    microbatch_size: int | None,
) -> int | jax.Array:
  """Calculates the number of non-padding microbatches.

  The returned  result is 1 + the index of the last microbatch that contains at
  least one non-padding example.  This means that microbatches consisting of
  all-padding examples that do not appear at the end will be treated as a real
  microbatch.

  Args:
    is_padding_example: A 1D array of shape (num_examples,).
    microbatch_size: Argument passed to `inmemory_microbatched_fn_general`.

  Returns:
    The `true` batch size, as a scalar jax array.
  """
  if microbatch_size is None:
    return is_padding_example.shape[0]
  reshaped = _sharding_aware_reshape(is_padding_example, microbatch_size)
  # Ensure there is at least one True in the array.
  is_real_batch = jnp.append(True, ~reshaped.all(axis=1))
  # We want the last real microbatch, argmax returns the first True value,
  # so we add increasing numbers from 0 to 1 to each index.
  return jnp.argmax(is_real_batch + jnp.linspace(0, 1, is_real_batch.size))


# pylint: disable=g-bare-generic
def inmemory_microbatched_fn_general(
    fun: Callable,
    batch_argnums: int | Sequence[int],
    microbatch_size: int | None,
    accumulation_type: Any = AccumulationType.SUM,
    dtype: jax.typing.DTypeLike | None = None,
) -> Callable:
  """A general microbatching transformation.

  Conceptually, given `fun`, this function returns a new function that does
  something like the following (for the case of SUM aggregation):
  ```
  def microbatched_fun(full_batch):
    accumulator = 0
    for microbatch in full_batch:
      accumulator += fun(microbatch)
    return accumulator
  ```
  where under the hood the `for` is implemented via a `lax.fori_loop` and hence
  forced to be sequential.

  This function is useful when evaluating `fun` on the full input batch exceeds
  available device memory. By splitting the batch into smaller microbatches and
  processing them sequentially, peak memory usage can be significantly reduced.
  Because the function is evaluated on smaller batches, this transformation
  requires knowledge of how the individual microbatch results should be combined
  back together (SUM, MEAN, or CONCAT). See the accumulation_type argument for
  more details. In an ideal world, the compiler would handle this memory-compute
  tradeoff automatically (at least in the case of the `fun` is actually a sum,
  mean, or concat over rows and this function computes the same value as `fun`).

  Example Usage:
    >>> fun = lambda x: (x+1, jnp.sum(3*x))
    >>> data = jnp.array([1, 2, 3, 4])
    >>> fun(data)
    (Array([2, 3, 4, 5], dtype=int32), Array(30, dtype=int32))
    >>> strategy = (AccumulationType.CONCAT, AccumulationType.SUM)
    >>> microbatched_fun = inmemory_microbatched_fn_general(
    ...    fun, batch_argnums=0, microbatch_size=2, accumulation_type=strategy
    ... )
    >>> microbatched_fun(data)
    (Array([2, 3, 4, 5], dtype=int32), Array(30, dtype=int32))

  [Optional] Advanced Usage (Early Stopping): If fun consumes a keyword argument
  with the name `is_padding_example`, this will be used to early stop the
  microbatching loop if possible.  This is useful to handle variable batch sizes
  without recompilation. If an element of is_padding_example is True, it is
  assumed that the contribution of that element to fun is an all-zeros pytree.
  For AccumulationType.CONCAT, we still return results with the same size as the
  batch axis, and for AccumulationType.MEAN we divide by this number as well.

  Performance considerations: If using the `is_padding_example` argument, it is
  important to make sure the examples appear in the correct order. For
  performance reasons in distributed environments w.r.t. communication/sharding,
  the elements [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with a microbatch size of 2 would
  be grouped and processed as: [0, 5], [1, 6], [2, 7], [3, 8], [4, 9].
  If elements 8 and 9 are padding examples, the elements should be reordered as
  [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] so that the microbatches would be
  [0, 1], [2, 3], [4, 5], [6, 7], [8, 9] and the last microbatch can be skipped
  if is_padding = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]. See
  `compute_early_stopping_order` to reorder your data in a manner that allows 
  you to leverage this.

  Args:
      fun: An arbitrary function.
      batch_argnums: A sequence of argument indices that have a batch axis. All
        kwargs are assumed to have a batch axis, similar to `jax.vmap`.
      microbatch_size: The number of rows in the overall batch used in each
        microbatch. Smaller values reduces memory overhead, but require more
        sequential computation. This must evenly divide the batch axis size of
        the batch arguments.
      accumulation_type: Specifies how to combine results from each microbatch;
        can be a single `AccumulationType`, a pytree matching the structure of
        `fun`'s output, with `AccumulationType` values at the leaves, or
        anything in between (i.e., a PyTree prefix of `fun`'s output`).
      dtype: Optional dtype for the microbatched function output.

  Returns:
      A new function that evaluates fun sequentially num_microbatches times on
        subsets of data. Consumes the same args and kwargs as `fun`.
  """
  if microbatch_size is None:
    return fun

  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)

  def microbatched_fun(*args, **kwargs):
    batch_args = [args[i] for i in batch_argnums]
    batch_size = jax.tree.leaves(batch_args)[0].shape[0]
    if batch_size % microbatch_size != 0:
      raise ValueError(f'{batch_size=} not divisible by {microbatch_size=}')
    num_microbatches = batch_size // microbatch_size
    reshaped_batch_args = _sharding_aware_reshape(batch_args, microbatch_size)
    reshaped_kwargs = _sharding_aware_reshape(kwargs, microbatch_size)

    def f(index):
      fetch = lambda arg: jax.tree.map(lambda x: x[index], arg)
      inputs = list(args)
      for i, arg in zip(batch_argnums, reshaped_batch_args):
        inputs[i] = fetch(arg)
      input_kwargs = {k: fetch(kwarg) for k, kwarg in reshaped_kwargs.items()}
      return fun(*inputs, **input_kwargs)

    accumulator = Accumulator(num_microbatches, accumulation_type, dtype)

    def body_fun(index, carry):
      return accumulator.update(carry, f(index), index)

    if kwargs.get('is_padding_example') is not None:
      # Early stop if given variable batch sizes without recompilation.
      # To realize performance benefits, padding examples should be concentrated
      # in the tail microbatches.
      num_microbatches = _calculate_num_real_microbatches(
          kwargs['is_padding_example'], microbatch_size
      )

    answer = jax.lax.fori_loop(
        1, num_microbatches, body_fun, accumulator.init(f(0))
    )

    return accumulator.finalize(answer)

  return microbatched_fun


def compute_early_stopping_order(
    batch_size: int,
    microbatch_size: int | None,
) -> np.ndarray:
  """Return an index permutation so data is processed in order w/ microbatching.

  This is a helper function to reorder data so that they get processed in the
  same order by `inmemory_microbatched_fn_general` as they would be processed
  without microbatching. This can be particularly helpful when the last elements
  of the batch are padding examples, in which case if they appear in the
  same microbatch we can avoid processing them.  This function is only useful
  if using the "is_padding_example" keyword argument with
  `inmemory_microbatched_fn_general`.

  Example Usage:
    >>> order = compute_early_stopping_order(batch_size=10, microbatch_size=2)
    >>> order
    array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])

  When permuting the input data to `inmemory_microbatched_fn_general` according
  to the above permutation, the examples will be split up into 5 microbatchs:
  [0, 1], [2, 3], [4, 5], [6, 7], [8, 9] and processed sequentially.

    >>> _sharding_aware_reshape(order, microbatch_size=2)
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])

  We can see how this is directly useful in the context of padding below.
  Because the last two microbatches consist of only padding examples,
  `inmemory_microbatched_fn_general` will skip them, saving compute.

    >>> is_padding = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    >>> _sharding_aware_reshape(is_padding[order], microbatch_size=2)
    array([[0, 0],
           [0, 0],
           [0, 0],
           [1, 1],
           [1, 1]])

  Args:
    batch_size: The size of the batch axis.
    microbatch_size: The target microbatch size that will be used with
      `inmemory_microbatched_fn_general`.

  Returns:
    A permutation of the example indices, where padding examples are evenly
    distributed across the microbatch indices and appear in the last k
    microbatches.  This is useful for early stopping when the true batch size is
    less than the size of the batch axis.
  """
  indices = np.arange(batch_size)
  if microbatch_size is None:
    return indices
  elif batch_size % microbatch_size != 0:
    raise ValueError(
        f'batch_size={batch_size} is not divisible by {microbatch_size=}'
    )
  return indices.reshape(-1, microbatch_size).T.flatten()


def verify_early_stopping_order(
    is_padding_example: jax.Array,
    microbatch_size: int | None,
) -> bool:
  """Verifies that is_padding gives the best early stopping with microbatching.

  Should not be called from a jitted context.

  Args:
    is_padding_example: A 1D array of shape (num_examples,).
    microbatch_size: Argument passed to `inmemory_microbatched_fn_general`.

  Returns:
    True if the is_padding_example gives an optimal early-stopping order.
  """
  microbatches = _sharding_aware_reshape(is_padding_example, microbatch_size)
  # all True values should be at the bottom of this matrix.
  num_padding_examples = microbatches.sum(axis=1)
  num_no_padding = (num_padding_examples == 0).sum()
  num_all_padding = (num_padding_examples == microbatch_size).sum()

  cond1 = num_no_padding + num_all_padding + 1 >= microbatches.shape[0]
  cond2 = (num_padding_examples[:num_no_padding] == 0).all()
  cond3 = (num_padding_examples[-num_all_padding:] == microbatch_size).all()
  cond3 = jax.lax.select(num_all_padding == 0, True, cond3)
  return cond1 & cond2 & cond3
