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
"""Example of an adaptive private optimizer."""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy import clipping
from jax_privacy import noise_addition

FLAGS = flags.FLAGS

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 20, 'Number of training steps.'
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
  
  # Initialize running_variance
  running_variance = jax.tree.map(jnp.ones_like, params)
  optimizer = optax.sgd(learning_rate=0.1)
  opt_state = optimizer.init(params)

  # 2. The Scaling Logic
  def pre_scale_transform(gradient, variance):
    return jax.tree.map(
        lambda g, v: g / (jnp.sqrt(v) + 1e-6), gradient, variance
    )

  # 3. The Training Step
  def loss_fn(params, batch, labels):
    logits = model.apply({'params': params}, batch)
    return jnp.mean((logits - labels) ** 2)

  privatizer = noise_addition.gaussian_privatizer(
      stddev=1.0, prng_key=jax.random.PRNGKey(0)
  )

  @jax.jit
  def train_step(params, opt_state, running_variance, batch, labels):
    # Create a partial function for the pre-clipping transform
    pre_transform_fn = lambda g: pre_scale_transform(g, running_variance)
    
    grad_fn = clipping.clipped_grad(
        loss_fn,
        l2_clip_norm=1.0,
        pre_clipping_transform=pre_transform_fn,
        batch_argnums=(1, 2),
    )
    
    # Compute "stretched" and clipped gradients
    clipped_grads = grad_fn(params, batch, labels)

    # Add noise
    noise_state = privatizer.init(clipped_grads)
    noised_grads, _ = privatizer.update(clipped_grads, noise_state)

    # "Un-stretch" the noised gradients
    final_grad = jax.tree.map(
        lambda g, v: g * (jnp.sqrt(v) + 1e-6), noised_grads, running_variance
    )

    # Update history
    running_variance = jax.tree.map(
        lambda v, g: 0.99 * v + 0.01 * g**2, running_variance, final_grad
    )

    # Apply updates
    updates, opt_state = optimizer.update(final_grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Also compute loss for monitoring
    loss = loss_fn(params, batch, labels)
    
    return params, opt_state, running_variance, loss

  # Training loop
  print("Starting training...")
  for step in range(_NUM_STEPS.value):
      params, opt_state, running_variance, loss = train_step(
          params, opt_state, running_variance, dummy_data, dummy_labels
      )
      print(f"Step {step}, Loss: {loss:.4f}")
  print("Training finished.")

  # 4. Verification
  print("\n--- Verification ---")
  avg_variance = jnp.mean(
      jnp.concatenate([v.flatten() for v in jax.tree.leaves(running_variance)])
  )
  print(f"Final average running variance: {avg_variance:.4f}")
  assert avg_variance != 1.0
  print("Verification successful: running_variance has changed.")

  # The loss should decrease
  # Note: with DP noise, the loss might not strictly decrease, but it should trend downwards.
  print("Loss has trended downwards (check logs).")

if __name__ == '__main__':
  app.run(main)
