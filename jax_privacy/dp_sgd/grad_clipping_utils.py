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

"""Utils for grad_clipping.py."""

import abc
from collections.abc import Mapping
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import typing as dp_types


class StateAccumulationStrategy(metaclass=abc.ABCMeta):
  """Strategy for accumulating network state.

  When used in "loop" mode, its semantics should be as if it were called as:
  ```
    state = initialize(previous_state, batch_size=n)
    for i in range(n):
      ..., new_state = training_update(inputs[i:i+1], ..., previous_state)
      state = accumulate(state, new_state, batch_size=n, i=i)
  ```

  When used in "vect" mode, its semantics should be as if it were called as:
  ```
    ..., new_states = jax.vmap(
        lambda x: training_update(jnp.expand_dims(x, 0), ..., previous_state),
        inputs)
    state = aggregate(previous_state, new_states)
  ```
  """

  @abc.abstractmethod
  def initialize(
      self,
      previous_state: chex.Array,
      *,
      batch_size: int,
  ) -> chex.Array:
    """Initialises an accumulation of a state tensor.

    Called by `LoopAccumulator.initialize`.

    Args:
      previous_state: Incoming network state tensor, e.g. from the previous
        training step.
      batch_size: Number of examples in the batch.
    Returns:
      Initial value of the running network state tensor.
    """

  @abc.abstractmethod
  def accumulate(
      self,
      state: chex.Array,
      new_state: chex.Array,
      *,
      batch_size: int,
      i: int,
  ) -> chex.Array:
    """Incorporates one example's update into the state tensor.

    Called by `LoopAccumulator.accumulate`.

    Args:
      state: Running network state tensor.
      new_state: Network state tensor arising from one example.
      batch_size: Number of examples in the batch.
      i: Index of this example within the batch.
    Returns:
      Updated value of the running network state tensor.
    """

  @abc.abstractmethod
  def aggregate(
      self,
      previous_state: chex.Array,
      new_states: chex.Array,
  ) -> chex.Array:
    """Aggregates examples' state tensor updates into a new state tensor.

    Called by `reduce_vmap`.

    Args:
      previous_state: Network state tensor in effect at the start of this batch.
      new_states: State tensor with an extra leading dimension, containing the
        states arising from each example.

    Returns:
      State tensor arising from the entire batch.
    """


class Reject(StateAccumulationStrategy):
  """State accumulator that asserts that there are no leaves in the state."""

  def initialize(
      self,
      previous_state: chex.Array,
      *,
      batch_size: int,
  ) -> chex.Array:
    raise ValueError('Unhandled network state')

  def accumulate(
      self,
      state: chex.Array,
      new_state: chex.Array,
      *,
      batch_size: int,
      i: int,
  ) -> chex.Array:
    raise ValueError('Unhandled network state')

  def aggregate(
      self,
      previous_state: chex.Array,
      new_states: chex.Array,
  ) -> chex.Array:
    raise ValueError('Unhandled network state')


class Average(StateAccumulationStrategy):
  """State accumulator that averages across microbatches."""

  def initialize(
      self,
      previous_state: chex.Array,
      *,
      batch_size: int,
  ) -> chex.Array:
    del batch_size
    return jnp.zeros_like(previous_state)

  def accumulate(
      self,
      state: chex.Array,
      new_state: chex.Array,
      *,
      batch_size: int,
      i: int,
  ) -> chex.Array:
    del i
    return state + new_state / batch_size

  def aggregate(
      self,
      previous_state: chex.Array,
      new_states: chex.Array,
  ) -> chex.Array:
    del previous_state
    return jnp.mean(new_states, axis=0)


class Sum(StateAccumulationStrategy):
  """State accumulator taking the "vector sum" of updates across microbatches.

  If the previous state is `p`, and the `i`th single-example update gives an
  updated state of `s_i`, then the resulting state is deemed to be:
  ```
    p + sum_i (s_i - p)
  ```

  This simplifies to:
  ```
    (1-n)p + sum_i s_i
  ```

  In loop mode, the `(1-p)` term is handled within `initialize()`.
  """

  def initialize(
      self,
      previous_state: chex.Array,
      *,
      batch_size: int,
  ) -> chex.Array:
    # Each batch will add a copy of the previous state during summing,
    # but we only want one copy of the previous state.
    # Correct for that in advance.
    return previous_state * (1. - batch_size)

  def accumulate(
      self,
      state: chex.Array,
      new_state: chex.Array,
      *,
      batch_size: int,
      i: int,
  ) -> chex.Array:
    del batch_size, i
    return state + new_state

  def aggregate(
      self,
      previous_state: chex.Array,
      new_states: chex.Array,
  ) -> chex.Array:
    return previous_state + jnp.sum(new_states - previous_state, axis=0)


StateAccumulationStrategyTree = (
    StateAccumulationStrategy | Mapping[str, 'StateAccumulationStrategyTree']
)


# Tree-mapped versions of StateAccumulationStrategy methods.
# Because a StateAccumulationStrategyTree is a prefix tree, each entry will in
# general act upon an "inner" sub-tree. These functions support that sub-tree
# action.


def _initialize_subtree(
    state_acc_strategy: StateAccumulationStrategy,
    previous_state: chex.ArrayTree,
    *,
    batch_size: int,
) -> chex.ArrayTree:
  return jax.tree_util.tree_map(
      functools.partial(state_acc_strategy.initialize, batch_size=batch_size),
      previous_state)


def _accumulate_subtree(
    state_acc_strategy: StateAccumulationStrategy,
    state: chex.Array,
    new_state: chex.Array,
    *,
    batch_size: int,
    i: int,
) -> chex.ArrayTree:
  return jax.tree_util.tree_map(
      functools.partial(
          state_acc_strategy.accumulate, batch_size=batch_size, i=i),
      state, new_state)


def _aggregate_subtree(
    state_acc_strategy: StateAccumulationStrategy,
    previous_state: chex.ArrayTree,
    new_states: chex.ArrayTree,
) -> chex.ArrayTree:
  return jax.tree_util.tree_map(
      state_acc_strategy.aggregate, previous_state, new_states)


_ValueAndGrad = tuple[
    tuple[dp_types.Loss, tuple[dp_types.ModelStateT, dp_types.Metrics]],
    dp_types.ParamsT,
]


class LoopAccumulator:
  """Accumulate or stack values and grads over a loop."""

  def __init__(
      self,
      value_and_grad_fn: dp_types.ValueAndGradFn,
      state_acc_strategies: StateAccumulationStrategyTree,
  ):
    self._value_and_grad_fn = value_and_grad_fn
    self._state_acc_strategies = state_acc_strategies

  def initialize(
      self,
      network_state: dp_types.ModelStateT,
      batch_size: int,
      *arg_shapes,
  ) -> _ValueAndGrad:
    """Initializes the scan loop."""
    loss_and_grad = jax.eval_shape(self._value_and_grad_fn, *arg_shapes)
    (loss, (unused_state, metrics)), grads = jax.tree_util.tree_map(
        jnp.zeros_like, loss_and_grad)
    metrics = metrics.replace(
        per_example={
            'grad_norm': jnp.zeros((batch_size,), dtype=loss.dtype),
            **metrics.per_example,
        },
        scalars_avg=jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32), metrics.scalars_avg)
    )

    # This `tree_map`` is over the "outer" (prefix) structure specified by
    # `self._state_acc_strategies`.
    # `tree_map` tolerates the first arg being a prefix tree of the later args,
    # so each invocation of `_initialize_tree` acts on an inner sub-tree.
    network_state = jax.tree_util.tree_map(
        functools.partial(_initialize_subtree, batch_size=batch_size),
        self._state_acc_strategies,
        network_state,
    )

    return (loss, (network_state, metrics)), grads

  def accumulate(
      self,
      value_and_grad: _ValueAndGrad,
      value_and_grad_i: _ValueAndGrad,
      batch_size: int,
      i: int,
  ) -> _ValueAndGrad:
    """Running average or stack of `value_and_grad_i` into `value_and_grad`."""

    def accumulate_mean(array: jax.Array, array_i: jax.Array) -> jax.Array:
      return (array + array_i / batch_size).astype(array.dtype)

    def accumulate_sum(array: jax.Array, array_i: jax.Array) -> jax.Array:
      return (array + array_i).astype(array.dtype)

    def update_at_i(array: jax.Array, array_i: jax.Array) -> jax.Array:
      return array.at[i].set(array_i.astype(array.dtype))

    (loss, (network_state, metrics)), grads = value_and_grad
    (loss_i, (state_i, metrics_i)), grads_i = value_and_grad_i
    loss = accumulate_mean(loss, loss_i)
    metrics_avg = jax.tree_util.tree_map(
        accumulate_mean, metrics.scalars_avg, metrics_i.scalars_avg)
    metrics_sum = jax.tree_util.tree_map(
        accumulate_sum, metrics.scalars_sum, metrics_i.scalars_sum)
    grads = jax.tree_util.tree_map(accumulate_mean, grads, grads_i)
    metrics_per_example = jax.tree_util.tree_map(
        update_at_i, metrics.per_example, metrics_i.per_example)
    metrics = dp_types.Metrics(
        scalars_avg=metrics_avg,
        scalars_sum=metrics_sum,
        per_example=metrics_per_example,
    )

    # This `tree_map` is over the "outer" (prefix) structure specified by
    # `self._state_acc_strategies`.
    # `tree_map` tolerates the first arg being a prefix tree of the later args,
    # so each invocation of `_accumulate_tree` acts on an inner sub-tree.
    network_state = jax.tree_util.tree_map(
        functools.partial(_accumulate_subtree, batch_size=batch_size, i=i),
        self._state_acc_strategies,
        network_state,
        state_i,
    )

    return (loss, (network_state, metrics)), grads


def reduce_vmap(
    state_acc_strategies: StateAccumulationStrategyTree,
    value_and_grads: _ValueAndGrad,
    previous_state: dp_types.ModelStateT,
) -> _ValueAndGrad:
  """Reduces the vmapped outputs."""
  (loss, (network_state, metrics)), grads = value_and_grads
  leaf_mean = functools.partial(jnp.mean, axis=0)
  tree_mean = functools.partial(jax.tree_util.tree_map, leaf_mean)
  leaf_sum = functools.partial(jnp.sum, axis=0)
  tree_sum = functools.partial(jax.tree_util.tree_map, leaf_sum)
  loss = tree_mean(loss)
  grads = tree_mean(grads)
  metrics = dataclasses.replace(
      metrics,
      scalars_avg=tree_mean(metrics.scalars_avg),
      per_example=metrics.per_example,
      scalars_sum=tree_sum(metrics.scalars_sum),
  )

  # This `tree_map`` is over the "outer" (prefix) structure specified by
  # `state_acc_strategies`.
  # `tree_map` tolerates the first arg being a prefix tree of the later args,
  # so each invocation of `_initialize_tree` acts on an inner sub-tree.
  network_state = jax.tree_util.tree_map(
      _aggregate_subtree, state_acc_strategies, previous_state, network_state)

  return (loss, (network_state, metrics)), grads
