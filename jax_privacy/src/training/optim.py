# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

from typing import Any, Callable, Mapping, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src import accounting
import optax


def l2_loss(params: chex.ArrayTree) -> chex.Array:
  """Computes the squared L2 loss.

  Args:
    params: model parameters for which the loss should be computed, assumed to
      be in haiku-like format. Layers whose name includes 'batchnorm' are
      ignored.
  Returns:
    Squared L2 loss.
  """
  l2_norm = 0.
  for mod_name, mod_params in params.items():
    if 'batchnorm' not in mod_name:
      l2_norm += sum(
          jnp.sum(jnp.square(param)) for param in mod_params.values())
  return 0.5 * l2_norm


def optimizer(
    *,
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, Any],
    learning_rate: float,
    every_k_schedule: Union[int, Callable[[chex.Array], chex.Array]],
) -> optax.GradientTransformation:
  """Create an optax optimizer that can accumulate gradients over steps.

  For example, if `every_k_schedule=2`, on the first step, the optimizer will
  store the gradient of the first mini-batch and not perform any update.
  On the second step, the optimizer will average the gradient of the second
  mini-batch with that of the first one, perform the update, and reset its
  memory, and so on. This allows to use large virtual batch-sizes.

  Args:
    optimizer_name: optax identifier for the optimizer.
    optimizer_kwargs: optax keyword arguments for the optimizer.
    learning_rate: learning-rate given to the optimizer.
    every_k_schedule: schedule (value or function) telling how often the
      optimizer should update the model parameters vs accumulating gradients
      to form a larger virtual batch-size.
  Returns:
    Optax optimizer.
  """
  constructor = getattr(optax, optimizer_name)
  opt = constructor(learning_rate, **optimizer_kwargs)
  opt_every_k = optax.MultiSteps(
      opt, every_k_schedule).gradient_transformation()
  return opt_every_k


def learning_rate_schedule(
    *,
    init_value: float,
    decay_schedule_name: Optional[str] = None,
    decay_schedule_kwargs: Optional[Mapping[str, Any]] = None,
    update_step: chex.Numeric,
) -> chex.Numeric:
  """Compute the learning-rate according to the given schedule.

  Args:
    init_value: initial value of the learning-rate schedule.
    decay_schedule_name: name of the optax schedule to use.
    decay_schedule_kwargs: keywordd arguments for the optax schedule.
    update_step: current update step that 'indexes' the schedule.
  Returns:
    Current value of the learning-rate.
  """
  if decay_schedule_name is not None:
    schedule_fn = getattr(optax, decay_schedule_name)
    if decay_schedule_kwargs is None:
      decay_schedule_kwargs = {}
    decay = schedule_fn(**decay_schedule_kwargs)(update_step)
  else:
    decay = 1.0
  return init_value * decay


def add_noise_to_grads(
    *,
    clipping_norm: Optional[chex.Numeric],
    rescale_to_unit_norm: bool,
    noise_std_relative: Optional[chex.Numeric],
    apply_every: int,
    total_batch_size: int,
    grads: chex.ArrayTree,
    rng_key: chex.PRNGKey,
) -> Tuple[chex.ArrayTree, chex.Array]:
  """Add noise to gradients.

  Args:
    clipping_norm: clipping-norm for the per-example gradients (before
      averaging across the examples of the mini-batch).
    rescale_to_unit_norm: whether each clipped per-example gradient gets
      multiplied by `1 / clipping_norm`, so that the update is normalized.
      When enabled, the noise standard deviation gets adjusted accordingly.
    noise_std_relative: standard deviation of the noise to add to the average
       of the clipped gradient to make it differentially private. It will be
       multiplied by `clipping_norm / total_batch_size` before the noise gets
       actually added.
    apply_every: how often the optimizer applies an actual update, i.e. for many
      steps gradients are accumulated (to grow the batch-size) before they get
      use to update the model parameters.
    total_batch_size: total batch-size once accumulated over devices and steps
      (i.e. as seen by the optimizer performing the update).
    grads: gradients to privatize.
    rng_key: random number generation key.
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

  if not noise_std_relative:
    # No noise to add (whether the clipping-norm is finite or not).
    total_noise_std = 0.0
  elif not clipping_norm_is_finite:
    # Cannot add noise proportional to infinity.
    raise ValueError(
        'noise_std_relative cannot be used without a finite clipping norm.')
  else:
    # The total amount of noise to add is the product of the scale and
    # noise_std_relative.
    assert noise_std_relative >= 0
    total_noise_std = scale * noise_std_relative

  # NB: no need to accumulate noise over devices because the noise is applied
  # identically on all devices
  std = accounting.divide_std_over_avg(total_noise_std, n=apply_every)
  noisy_grads = tree_map_add_normal_noise(grads, std, rng_key)
  return noisy_grads, std


def cosine_distance(
    tree_1: chex.ArrayTree,
    tree_2: chex.ArrayTree,
) -> chex.Array:
  """Compute cosine distance between two trees of arrays."""
  dot_product = sum(jax.tree_leaves(jax.tree_map(
      lambda g1, g2: jnp.sum(g1 * g2), tree_1, tree_2)))

  return dot_product / (optax.global_norm(tree_1) * optax.global_norm(tree_2))


def tree_map_add_normal_noise(
    tree: chex.ArrayTree,
    noise_std: float,
    rng_key: chex.PRNGKey,
) -> chex.ArrayTree:
  """Add iid gaussian noise with std 'noise_std' to all leaves of 'tree'."""
  rng_keys = jax.random.split(rng_key, len(jax.tree_leaves(tree)))
  rng_tree = jax.tree_unflatten(jax.tree_structure(tree), rng_keys)

  return jax.tree_map(
      lambda rng, x: x + noise_std * jax.random.normal(rng, shape=x.shape),
      rng_tree, tree)
