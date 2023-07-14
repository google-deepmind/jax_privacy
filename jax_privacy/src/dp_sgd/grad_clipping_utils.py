# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

import functools

import jax
import jax.numpy as jnp

from jax_privacy.src.dp_sgd import typing


_ValueAndGrad = tuple[
    tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
    typing.ParamsT,
]


class LoopAccumulator:
  """Accumulate or stack values and grads over a loop."""

  def __init__(self, value_and_grad_fn: typing.ValueAndGradFn):
    self._value_and_grad_fn = value_and_grad_fn

  def initialize(
      self,
      batch_size: int,
      *arg_shapes,
  ) -> _ValueAndGrad:
    """Initializes the scan loop."""
    loss_and_grad = jax.eval_shape(self._value_and_grad_fn, *arg_shapes)
    (loss, (network_state, metrics)), grads = jax.tree_util.tree_map(
        jnp.zeros_like, loss_and_grad)
    metrics = metrics.replace(
        per_example={
            'grad_norm': jnp.zeros((batch_size,), dtype=loss.dtype),
            **metrics.per_example,
        })
    return (loss, (network_state, metrics)), grads

  def accumulate(
      self,
      value_and_grad: _ValueAndGrad,
      value_and_grad_i: _ValueAndGrad,
      i: int,
      batch_size: int,
  ) -> _ValueAndGrad:
    """Running average or stack of `value_and_grad_i` into `value_and_grad`."""

    def accumulate_mean(array: jax.Array, array_i: jax.Array) -> jax.Array:
      return array + array_i / batch_size

    def accumulate_sum(array: jax.Array, array_i: jax.Array) -> jax.Array:
      return array + array_i

    def update_at_i(array: jax.Array, array_i: jax.Array) -> jax.Array:
      array_i = jnp.reshape(array_i, array[i].shape)
      return array.at[i].set(array_i)

    (loss, (unused_state, metrics)), grads = value_and_grad
    (loss_i, (state_i, metrics_i)), grads_i = value_and_grad_i
    loss = accumulate_mean(loss, loss_i)
    metrics_avg = jax.tree_map(
        accumulate_mean, metrics.scalars_avg, metrics_i.scalars_avg)
    metrics_sum = jax.tree_map(
        accumulate_sum, metrics.scalars_sum, metrics_i.scalars_sum)
    grads = jax.tree_map(accumulate_mean, grads, grads_i)
    metrics_per_example = jax.tree_map(
        update_at_i, metrics.per_example, metrics_i.per_example)
    metrics = typing.Metrics(
        scalars_avg=metrics_avg,
        scalars_sum=metrics_sum,
        per_example=metrics_per_example,
    )
    return (loss, (state_i, metrics)), grads


def reduce_vmap(
    value_and_grads: _ValueAndGrad,
) -> tuple[
    tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
    typing.ParamsT,
]:
  """Reduces the vmapped outputs."""
  (loss, (network_state, metrics)), grads = value_and_grads
  tree_mean = (
      lambda tree: jax.tree_map(functools.partial(jnp.mean, axis=0), tree))
  tree_sum = (
      lambda tree: jax.tree_map(functools.partial(jnp.sum, axis=0), tree))
  tree_squeeze = lambda tree: jax.tree_map(jnp.squeeze, tree)
  loss = tree_mean(loss)
  grads = tree_mean(grads)
  metrics = metrics.replace(
      scalars_avg=tree_mean(metrics.scalars_avg),
      per_example=tree_squeeze(metrics.per_example),
      scalars_sum=tree_sum(metrics.scalars_sum),
  )
  return (loss, (network_state, metrics)), grads
