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

"""Optim utils."""

import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import typing


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
    clipping_norm: chex.Numeric | None,
    rescale_to_unit_norm: bool,
    noise_multiplier: chex.Numeric | None,
    total_batch_size: chex.Numeric,
    grads: typing.ParamsT,
    rng_per_batch: chex.PRNGKey,
) -> tuple[typing.ParamsT, jax.Array]:
  """Add noise to gradients.

  Args:
    clipping_norm: clipping-norm for the per-example gradients (before averaging
      across the examples of the mini-batch).
    rescale_to_unit_norm: whether each clipped per-example gradient gets
      multiplied by `1 / clipping_norm`, so that the update is normalized. When
      enabled, the noise standard deviation gets adjusted accordingly.
    noise_multiplier: standard deviation of the noise to add to the average of
      the clipped gradient to make it differentially private. It will be
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
  clipping_scale = compute_clipping_norm_scale(
      clipping_norm, rescale_to_unit_norm, total_batch_size
  )
  noise_std = compute_noise_std(noise_multiplier, clipping_scale)
  noisy_grads = tree_map_add_normal_noise(grads, noise_std, rng_per_batch)
  return noisy_grads, noise_std


def compute_clipping_norm_scale(
    clipping_norm: chex.Numeric | None,
    rescale_to_unit_norm: bool,
    total_batch_size: chex.Numeric,
) -> chex.Numeric | None:
  """Computes the scalar multiplicative factor for clipping.

  Args:
    clipping_norm: clipping-norm for the per-example gradients (before averaging
      across the examples of the mini-batch).
    rescale_to_unit_norm: whether each clipped per-example gradient gets
      multiplied by `1 / clipping_norm`, so that the update is normalized. When
      enabled, the noise standard deviation gets adjusted accordingly.
    total_batch_size: total batch-size once accumulated over devices and steps
      (i.e. as seen by the optimizer performing the update).

  Returns:
    A float representing the computed scalar multiplicative factor.
  """
  if clipping_norm in (None, float('inf')):
    scale = None
  elif rescale_to_unit_norm:
    scale = 1.0 / jnp.asarray(total_batch_size, dtype=jnp.float32)
  else:
    scale = clipping_norm / jnp.asarray(total_batch_size, dtype=jnp.float32)
  return scale


def compute_noise_std(
    noise_multiplier: chex.Numeric | None,
    clipping_scale: chex.Numeric | None,
) -> chex.Numeric:
  """Computes the standard deviation for Gaussian noise.

  Args:
    noise_multiplier: standard deviation of the noise to add to the average of
      the clipped gradient to make it differentially private.
    clipping_scale: scalar factor to multiply with the noise multiplier to
      achieve the noise standard deviation.

  Returns:
    The noise standard deviation.

  Raises:
    ValueError: If `clipping_norm_is_finite` is False, i.e., the clipping norm
      is not finite.
  """
  if clipping_scale is None:
    clipping_norm_is_finite = False
  elif clipping_scale is float('inf'):
    clipping_norm_is_finite = False
  else:
    clipping_norm_is_finite = True

  if not noise_multiplier:
    # No noise to add (whether the clipping-norm is finite or not).
    return jnp.array(0.0)
  elif not clipping_norm_is_finite:
    # Cannot add noise proportional to infinity.
    raise ValueError(
        'noise_multiplier cannot be used without a finite clipping norm.'
    )
  else:
    return jnp.asarray(clipping_scale * noise_multiplier)


def tree_map_add_normal_noise(
    tree: typing.ParamsT,
    noise_std: chex.Numeric,
    rng_key: chex.PRNGKey,
) -> typing.ParamsT:
  """Add iid gaussian noise with std 'noise_std' to all leaves of 'tree'."""
  rng_keys = jax.random.split(rng_key, len(jax.tree_util.tree_leaves(tree)))
  rng_tree = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(tree), rng_keys
  )

  def with_noise(rng: chex.Array, x: chex.Array) -> chex.Array:
    x_dtype = x.dtype
    x = x.astype(jnp.float32)
    scale = jnp.asarray(noise_std, dtype=jnp.float32)
    x_with_noise = x + scale * jax.random.normal(
        rng, shape=x.shape, dtype=jnp.float32
    )
    return x_with_noise.astype(x_dtype)

  return jax.tree_util.tree_map(with_noise, rng_tree, tree)
