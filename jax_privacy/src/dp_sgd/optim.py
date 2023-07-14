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

"""Optim utils."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import typing
import optax


def apply_weight_decay(
    tree: chex.ArrayTree,
    *,
    learning_rate: chex.Numeric,
    weight_decay: chex.Numeric,
) -> chex.ArrayTree:
  factor = 1.0 - learning_rate * weight_decay
  return jax.tree_util.tree_map(lambda x: factor * x, tree)


def add_noise_to_grads(
    *,
    clipping_norm: Optional[chex.Numeric],
    rescale_to_unit_norm: bool,
    noise_multiplier: Optional[chex.Numeric],
    total_batch_size: int,
    grads: typing.ParamsT,
    rng_per_batch: chex.PRNGKey,
) -> tuple[typing.ParamsT, chex.Numeric]:
  """Add noise to gradients.

  Args:
    clipping_norm: clipping-norm for the per-example gradients (before
      averaging across the examples of the mini-batch).
    rescale_to_unit_norm: whether each clipped per-example gradient gets
      multiplied by `1 / clipping_norm`, so that the update is normalized.
      When enabled, the noise standard deviation gets adjusted accordingly.
    noise_multiplier: standard deviation of the noise to add to the average
       of the clipped gradient to make it differentially private. It will be
       multiplied by `clipping_norm / total_batch_size` before the noise gets
       actually added.
    total_batch_size: total batch-size once accumulated over devices and steps
      (i.e. as seen by the optimizer performing the update).
    grads: gradients to privatize.
    rng_per_batch: random number generation key.
  Returns:
    noisy_grads: gradients with the added noise.
    std: standard deviation used for the noise (for monitoring purposes).
  """
  if clipping_norm in (None, float('inf')):
    clipping_norm_is_finite = False
    scale = None
  elif rescale_to_unit_norm:
    clipping_norm_is_finite = True
    scale = 1.0 / total_batch_size
  else:
    clipping_norm_is_finite = True
    scale = clipping_norm / total_batch_size

  if not noise_multiplier:
    # No noise to add (whether the clipping-norm is finite or not).
    std = 0.0
  elif not clipping_norm_is_finite:
    # Cannot add noise proportional to infinity.
    raise ValueError(
        'noise_multiplier cannot be used without a finite clipping norm.')
  else:
    # The total amount of noise to add is the product of the scale and
    # noise_multiplier.
    assert noise_multiplier >= 0
    std = scale * noise_multiplier

  # NB: no need to accumulate noise over devices because the noise is applied
  # identically on all devices
  noisy_grads = tree_map_add_normal_noise(grads, std, rng_per_batch)
  return noisy_grads, std


def cosine_distance(
    tree_1: chex.ArrayTree,
    tree_2: chex.ArrayTree,
) -> chex.Array:
  """Compute cosine distance between two trees of arrays."""
  dot_product = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
      lambda g1, g2: jnp.sum(g1 * g2), tree_1, tree_2)))

  return dot_product / (optax.global_norm(tree_1) * optax.global_norm(tree_2))


def tree_map_add_normal_noise(
    tree: typing.ParamsT,
    noise_std: float,
    rng_key: chex.PRNGKey,
) -> typing.ParamsT:
  """Add iid gaussian noise with std 'noise_std' to all leaves of 'tree'."""
  rng_keys = jax.random.split(rng_key, len(jax.tree_util.tree_leaves(tree)))
  rng_tree = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(tree), rng_keys)

  def with_noise(rng: chex.Array, x: chex.Array) -> chex.Array:
    return x + noise_std * jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
  return jax.tree_util.tree_map(with_noise, rng_tree, tree)
