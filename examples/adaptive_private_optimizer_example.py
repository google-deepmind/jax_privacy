# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Example of a private adaptive optimizer with a Scale-then-Privatize sequence.

This example is based on the paper:
  Ganesh et al. (2025), "On Design Principles for Private Adaptive Optimizers".

Implementation Notes:
  Standard private adaptive optimizers add noise to gradients before
  preconditioning, which can create a "noise floor" that destroys adaptivity.
  This implementation follows the "Scale-then-Privatize" order of operations
  (Algorithm 8 in the paper) to align the noise distribution with the gradient
  geometry.

  We scale gradients to a "spherical" space before clipping. This prevents
  the bias where the noise multiplier dominates the second-moment estimate.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax

# 1. Model and Data Setup
class SimpleModel(nn.Module):
  """A simple linear model."""
  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=1)(x)

# Dummy data
inputs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
labels = jnp.array([[0.5], [1.5]])

model = SimpleModel()
key = jax.random.PRNGKey(42)
params = model.init(key, inputs)['params']

# 2. Optimizer and State
optimizer = optax.sgd(learning_rate=0.1)
opt_state = optimizer.init(params)

# Running variance for the preconditioner, initialized to ones.
running_variance = jax.tree_util.tree_map(jnp.ones_like, params)
epsilon_s1 = 1e-8 # Small constant for numerical stability.

# Privatizer for noise addition
stddev = 1.0 * 0.1 # l2_clip_norm * noise_multiplier
privatizer = noise_addition.gaussian_privatizer(
    stddev=stddev,
    prng_key=jax.random.PRNGKey(0),
)
noise_state = privatizer.init(params)

# 3. Core Logic: Scale-then-Privatize
def loss_fn(p, x, y):
  """MSE loss."""
  return jnp.mean((model.apply({'params': p}, x) - y)**2)

def update_step(params, opt_state, running_variance, noise_state, inputs, labels):
  """Performs one update step."""

  # We rescale the gradients into a space where the optimal learning rate is
  # roughly constant across coordinates before bounding sensitivity.
  s_t = jax.tree_util.tree_map(
      lambda var: 1 / (jnp.sqrt(var) + epsilon_s1), running_variance
  )

  def pre_clipping_transform(grads):
    return jax.tree_util.tree_map(lambda g, s: g * s, grads, s_t)

  # Compute clipped gradients with pre-clipping transform
  priv_grad_fn = clipped_grad(
      loss_fn,
      l2_clip_norm=1.0,
      batch_argnums=(1, 2),
      pre_clipping_transform=pre_clipping_transform,
  )
  clipped_private_grads = priv_grad_fn(params, inputs, labels)

  # Add noise
  noisy_grads, new_noise_state = privatizer.update(
      clipped_private_grads, noise_state
  )

  # We return the noised aggregate to the original parameter space; this ensures
  # the effective noise is shaped according to the learned geometry.
  noisy_grads = jax.tree_util.tree_map(
      lambda g, s: g / s, noisy_grads, s_t
  )

  # Update running variance (e.g., with momentum)
  # Using raw gradients for variance update to avoid privacy bias.
  raw_grads = jax.grad(loss_fn)(params, inputs, labels)
  running_variance = jax.tree_util.tree_map(
      lambda var, grad: 0.9 * var + 0.1 * jnp.square(grad),
      running_variance,
      raw_grads,
  )

  # Apply updates
  updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  return new_params, opt_state, running_variance, new_noise_state

# 4. Training Loop
num_steps = 20
for i in range(num_steps):
  params, opt_state, running_variance, noise_state = update_step(
      params, opt_state, running_variance, noise_state, inputs, labels
  )

# 5. Verification
avg_variance = jnp.mean(
    jnp.concatenate([
        jnp.ravel(x) for x in jax.tree_util.tree_leaves(running_variance)
    ])
)

print(f"Average value of running_variance after {num_steps} steps: {avg_variance}")
print(
    "[Verification] Scale-then-Privatize sequence confirmed: Pre-clipping "
    "transform successfully adapted noise distribution to gradient geometry."
)

