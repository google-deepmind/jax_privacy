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

"""Accumulates the metrics over multiple steps."""

from collections.abc import Mapping
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import typing


def _maybe_expand(x: chex.Array, axis: int) -> chex.Array:
  """Returns an array expanded until dimension `axis`."""
  if x.ndim <= axis:
    for a in range(x.ndim, axis + 1):
      x = jnp.expand_dims(x, axis=a)
  return x


def _concatenate_arrays(
    x1: chex.Array,
    x2: chex.Array,
    axis: int,
) -> chex.Array:
  """Returns the concatenation of two arrays along `axis`."""
  x1 = _maybe_expand(x1, axis)
  x2 = _maybe_expand(x2, axis)
  return jnp.concatenate([x1, x2], axis=axis)


def _concatenate(
    tree_1: Mapping[str, chex.Numeric],
    tree_2: Mapping[str, chex.Numeric],
    axis: int,
) -> Mapping[str, chex.Numeric]:
  """Returns the concatenation of two dictionaries along `axis`."""
  if not tree_1:
    return tree_2
  elif not tree_2:
    return tree_1
  else:
    return jax.tree_util.tree_map(
        lambda x1, x2: _concatenate_arrays(x1, x2, axis=axis),
        tree_1, tree_2)


def _avg(
    tree_1: Mapping[str, chex.Numeric],
    tree_2: Mapping[str, chex.Numeric],
    axis: int,
    count_1: int,
    count_2: int,
) -> Mapping[str, chex.Numeric]:
  """Returns the weighted average of two dictionaries along `axis`."""
  total_count = count_1 + count_2
  tree_1 = jax.tree_util.tree_map(lambda x: x*count_1/total_count, tree_1)
  tree_2 = jax.tree_util.tree_map(lambda x: x*count_2/total_count, tree_2)
  return _add(tree_1, tree_2, axis=axis)


def _add(
    tree_1: Mapping[str, chex.Numeric],
    tree_2: Mapping[str, chex.Numeric],
    axis: int,
) -> Mapping[str, chex.Numeric]:
  """Returns the sum of two dictionaries along `axis`."""
  if not tree_1:
    return tree_2
  elif not tree_2:
    return tree_1
  else:
    return jax.tree_util.tree_map(
        functools.partial(jnp.sum, axis=axis),
        _concatenate(tree_1, tree_2, axis=axis),
    )


@dataclasses.dataclass
class MetricsAccumulator:
  """Accumulates metrics."""

  inner: typing.Metrics = dataclasses.field(default_factory=typing.Metrics)
  count: chex.Numeric = 0
  axis: int = 1

  def accumulate(
      self,
      other: typing.Metrics,
      other_count: chex.Numeric,
  ) -> 'MetricsAccumulator':
    """Accumulate inner metrics with other."""
    scalars_avg = _avg(
        self.inner.scalars_avg,
        other.scalars_avg,
        axis=self.axis,
        count_1=self.count,
        count_2=other_count,
    )
    scalars_sum = _add(
        self.inner.scalars_sum,
        other.scalars_sum,
        axis=self.axis,
    )
    per_example = _concatenate(
        self.inner.per_example,
        other.per_example,
        axis=self.axis,
    )
    return MetricsAccumulator(
        inner=typing.Metrics(
            scalars_avg=scalars_avg,
            scalars_sum=scalars_sum,
            scalars_last=other.scalars_last,
            per_example=per_example,
        ),
        count=self.count+other_count,
        axis=self.axis,
    )

  def to_metrics(self) -> typing.Metrics:
    """Returns the accumulated metrics in `typing.Metrics` format."""
    scalars_last = {**self.inner.scalars_last}
    for name, values in self.inner.per_example.items():
      scalars_last.update({
          f'{name}_mean': jnp.mean(values, axis=self.axis),
          f'{name}_min': jnp.min(values, axis=self.axis),
          f'{name}_max': jnp.max(values, axis=self.axis),
          f'{name}_std': jnp.std(values, axis=self.axis),
          f'{name}_median': jnp.median(values, axis=self.axis),
      })
    return typing.Metrics(
        scalars_avg={**self.inner.scalars_avg},
        scalars_sum={**self.inner.scalars_sum},
        scalars_last=scalars_last,
        per_example={**self.inner.per_example},
    )
