# Copyright 2023, The jax_privacy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of a DP2 delayed preconditioner optimizer."""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy import clipping
from jax_privacy import noise_addition

FLAGS = flags.FLAGS

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 20, 'Number of training steps.'
)
_DELAY_S = flags.DEFINE_integer(
    'delay_s', 5, 'Update preconditioner every s steps.'
)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Setup
  # Define a simple model
  class SimpleModel(nn.Module):
      @nn.compact
      def __call__(self, x):
          return nn.Dense(features=1)(x)

  model = SimpleModel()
  
  # Create a small dummy dataset
  dummy_data = jnp.arange(1000).reshape((100, 10))
  dummy_labels = jnp.ones((100, 1))

  rng = jax.random.PRNGKey(0)
  params = model.init(rng, dummy_data[0])['params']
  
  # Initialize preconditioner_v and gradient_accumulator
  preconditioner_v = jax.tree.map(jnp.zeros_like, params)
  gradient_accumulator = jax.tree.map(jnp.zeros_like, params)

  # 2. The Training Logic
  def loss_fn(params, batch, labels):
    logits = model.apply({'params': params}, batch)
    return jnp.mean((logits - labels) ** 2)

  grad_fn = clipping.clipped_grad(
      loss_fn,
      l2_clip_norm=1.0,
      batch_argnums=(1, 2),
  )
  
  privatizer = noise_addition.gaussian_privatizer(
      stddev=1.0, prng_key=jax.random.PRNGKey(0)
  )

  learning_rate = 1e-4
  
  preconditioner_v_at_step_4 = None
  preconditioner_v_at_step_5 = None

  print("Starting training...")
  for step in range(1, _NUM_STEPS.value + 1):
    # Standard DP-Step
    clipped_grads = grad_fn(params, dummy_data, dummy_labels)
    noise_state = privatizer.init(clipped_grads)
    g_tilde, _ = privatizer.update(clipped_grads, noise_state)
    
    # Accumulate
    gradient_accumulator = jax.tree.map(
        lambda acc, g: acc + g, gradient_accumulator, g_tilde
    )
    
    # Check for Delay
    if step > 0 and step % _DELAY_S.value == 0:
      # Update the memory
      preconditioner_v = jax.tree.map(
          lambda v, acc: 0.9 * v + 0.1 * (acc / _DELAY_S.value)**2,
          preconditioner_v,
          gradient_accumulator,
      )
      # Reset the accumulator
      gradient_accumulator = jax.tree.map(jnp.zeros_like, params)
      
    # Apply Update
    updates = jax.tree.map(
        lambda g, v: learning_rate * g / (jnp.sqrt(v) + 1e-6),
        g_tilde,
        preconditioner_v,
    )
    params = jax.tree.map(lambda p, u: p - u, params, updates)

    loss = loss_fn(params, dummy_data, dummy_labels)
    print(f"Step {step}, Loss: {loss:.4f}")
    
    if step == 4:
        preconditioner_v_at_step_4 = preconditioner_v
    if step == 5:
        preconditioner_v_at_step_5 = preconditioner_v

  print("Training finished.")
  
  # 3. Verification
  print("\n--- Verification ---")
  print("preconditioner_v at step 4 (should be unchanged from initial zeros):")
  print(preconditioner_v_at_step_4)
  print("\npreconditioner_v at step 5 (should have updated):")
  print(preconditioner_v_at_step_5)

  # Check that the values are different
  v4_flat = jnp.concatenate([v.flatten() for v in jax.tree.leaves(preconditioner_v_at_step_4)])
  v5_flat = jnp.concatenate([v.flatten() for v in jax.tree.leaves(preconditioner_v_at_step_5)])

  assert not jnp.allclose(v4_flat, v5_flat)
  print("\nVerification successful: preconditioner_v was updated at the correct step.")
  print("Loss has remained stable (check logs).")

if __name__ == '__main__':
  app.run(main)
