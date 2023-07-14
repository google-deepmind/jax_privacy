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

"""Parameter averaging functions."""

import chex
import jax
import jax.numpy as jnp

from jax_privacy.src.dp_sgd import typing


def polyak(
    tree_old: typing.ParamsT,
    tree_new: typing.ParamsT,
    t: chex.Numeric,
) -> typing.ParamsT:
  """Polyak averaging if t >= 0, return tree_new otherwise."""
  t = jnp.maximum(t, 0)
  return jax.tree_util.tree_map(
      lambda old, new: (t * old + new) / (t + 1),
      tree_old,
      tree_new,
  )


def ema(
    tree_old: typing.ParamsT,
    tree_new: typing.ParamsT,
    mu: chex.Numeric,
    t: chex.Numeric,
    use_warmup: bool = True,
) -> typing.ParamsT:
  """Exponential Moving Averaging if t >= 0, return tree_new otherwise."""
  # Do not average until t >= 0.
  mu *= (t >= 0)
  if use_warmup:
    mu = jnp.minimum(mu, (1.0 + t) / (10.0 + t))
  return jax.tree_util.tree_map(
      lambda old, new: mu * old + (1 - mu) * new,
      tree_old,
      tree_new,
  )
