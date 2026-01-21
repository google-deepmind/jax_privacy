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
"""Example of a private adaptive optimizer with a Delayed Preconditioner (DP^2).

This example is based on the paper:
  Li et al. (2023), "Differentially Private Adaptive Optimization with Delayed
  Preconditioners" (ICLR 2023).

Implementation Notes:
  This implementation demonstrates the "alternating-phase protocol" of DP^2.
  The core idea is to introduce a trade-off between the staleness of the
  preconditioner and noise reduction. The preconditioner (second-moment
  estimate) is kept "stale" for a fixed number of steps (`delay_s`). During
  this period, noised gradients are accumulated.

  When the delay period is over, the preconditioner is updated using the
  *average* of the accumulated gradients. By averaging over multiple steps,
  the signal from the gradients is amplified relative to the noise, leading
  to a more stable and reliable preconditioner update. This helps to prevent
  the noise from overwhelming the adaptive learning rate calculation, a common
  problem in standard private adaptive optimizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax
import numpy as np

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
epsilon_s1 = 1e-8

# 2. Optimizer and DP^2 State
optimizer = optax.sgd(learning_rate=0.1)
opt_state = optimizer.init(params)

# DP^2 parameters
delay_s = 5
preconditioner_v = jax.tree_util.tree_map(jnp.ones_like, params)
gradient_accumulator = jax.tree_util.tree_map(jnp.zeros_like, params)

# Privatizer for noise addition
stddev = 1.0 * 0.1 # l2_clip_norm * noise_multiplier
privatizer = noise_addition.gaussian_privatizer(
    stddev=stddev,
    prng_key=jax.random.PRNGKey(0),
)
noise_state = privatizer.init(params)

# 3. Core Logic: DP^2 Update Step
def loss_fn(p, x, y):
  """MSE loss."""
  return jnp.mean((model.apply({'params': p}, x) - y)**2)

def update_step(
    step_idx, params, opt_state, preconditioner_v, gradient_accumulator, noise_state
):
  """Performs one update step with delayed preconditioning."""
  
  # Scale gradients using the (potentially stale) preconditioner
  s_t = jax.tree_util.tree_map(
      lambda var: 1 / (jnp.sqrt(var) + epsilon_s1), preconditioner_v
  )
  def pre_clipping_transform(grads):
    return jax.tree_util.tree_map(lambda g, s: g * s, grads, s_t)

  # Compute and clip gradients
  priv_grad_fn = clipped_grad(
      loss_fn,
      l2_clip_norm=1.0,
      batch_argnums=(1, 2),
      pre_clipping_transform=pre_clipping_transform,
  )
  clipped_grads = priv_grad_fn(params, inputs, labels)

  # Add noise
  noised_grads, new_noise_state = privatizer.update(
      clipped_grads, noise_state
  )
  
  # Rescale noised gradients back to original space
  noised_grads_rescaled = jax.tree_util.tree_map(
      lambda g, s: g / s, noised_grads, s_t
  )

  # Apply updates to parameters
  updates, new_opt_state = optimizer.update(noised_grads_rescaled, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  # --- DP^2 Preconditioner Update Logic ---
  # Accumulate the noised (but not rescaled) gradients
  new_gradient_accumulator = jax.tree_util.tree_map(
      lambda acc, g: acc + g, gradient_accumulator, noised_grads
  )

  def update_preconditioner(operand):
    """Update preconditioner with averaged gradients and reset accumulator."""
    p_v, grad_acc = operand
    # By delaying the update, we ensure that the second-moment estimate is
    # derived from a higher signal-to-noise ratio average, preventing the
    # noise from dominating the adaptive learning rates.
    avg_grad = jax.tree_util.tree_map(lambda x: x / delay_s, grad_acc)
    new_p_v = jax.tree_util.tree_map(
        lambda v, g: 0.9 * v + 0.1 * jnp.square(g), p_v, avg_grad
    )
    new_grad_acc = jax.tree_util.tree_map(jnp.zeros_like, grad_acc)
    return new_p_v, new_grad_acc

  def keep_stale(operand):
    """Keep preconditioner stale and continue accumulating."""
    return operand

  # "Snap" update: conditionally update the preconditioner
  new_preconditioner_v, new_gradient_accumulator = jax.lax.cond(
      (step_idx + 1) % delay_s == 0,
      update_preconditioner,
      keep_stale,
      (preconditioner_v, new_gradient_accumulator),
  )

  return (
      new_params,
      new_opt_state,
      new_preconditioner_v,
      new_gradient_accumulator,
      new_noise_state,
  )

# 4. Training Loop & Verification
v_initial = preconditioner_v
snapshots = {}
num_steps = 20

for i in range(num_steps):
  (
      params,
      opt_state,
      preconditioner_v,
      gradient_accumulator,
      noise_state,
  ) = update_step(
      i, params, opt_state, preconditioner_v, gradient_accumulator, noise_state
  )
  if i == 3: # Step 4 (0-indexed)
    snapshots['v_step4'] = preconditioner_v
  if i == 4: # Step 5 (0-indexed)
    snapshots['v_step5'] = preconditioner_v

# 5. Correctness Verification
def flatten_pytree(pt):
  return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(pt)])

v_initial_flat = flatten_pytree(v_initial)
v_step4_flat = flatten_pytree(snapshots['v_step4'])
v_step5_flat = flatten_pytree(snapshots['v_step5'])

# Assert that the preconditioner was stale until the update step
np.testing.assert_allclose(v_initial_flat, v_step4_flat, rtol=1e-6)
assert not np.allclose(v_step4_flat, v_step5_flat), "Preconditioner did not update at step 5."

print(
    "[Verification] DP2 Protocol confirmed: Preconditioner remained stale for"
    " steps 1-4 and updated successfully at step 5."
)
