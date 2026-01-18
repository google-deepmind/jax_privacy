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

"""Example of training with cryptographically secure noise.

This script demonstrates how to train a simple model with cryptographically
secure noise generated on the CPU and injected into the training step on the
accelerator.
"""

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from randomgen.aes import AESCounter
from randomgen.generator import ExtendedGenerator

from jax_privacy.batch_selection import CyclicPoissonSampling
from jax_privacy.noise_addition import gaussian_privatizer


def generate_secure_noise(params_tree, stddev, generator):
  """Generates a PyTree of Gaussian noise on the CPU."""
  numpy_generator = np.random.default_rng(generator.bit_generator)
  noise_tree = jax.tree.map(
      lambda x: stddev
      * numpy_generator.standard_normal(size=x.shape).astype(x.dtype),
      params_tree,
  )
  return noise_tree


class SimpleModel(nn.Module):
  """A simple linear model."""

  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=1)(x)


def loss_fn(params, model, batch):
  """Calculates the loss for a batch."""
  preds = model.apply({'params': params}, batch['image'])
  return jnp.mean((preds - batch['label']) ** 2)

def main():
  """Trains a simple model with secure noise."""
  # RNG Setup
  secure_rng = ExtendedGenerator(AESCounter())
  jax_rng = jax.random.PRNGKey(0)

  # Model and data setup
  model = SimpleModel()
  dummy_input = jnp.zeros((1, 28 * 28))
  params = model.init(jax_rng, dummy_input)['params']
  dataset_size = 1000
  batch_size = 64
  num_features = 28 * 28

  # Create a dummy dataset
  dummy_dataset = {
      'image': jnp.ones((dataset_size, num_features)),
      'label': jnp.ones((dataset_size, 1)),
  }

  # Training setup
  learning_rate = 0.1
  stddev = 1.0
  iterations = 20
  privatizer = gaussian_privatizer(stddev=stddev)
  optimizer = optax.sgd(learning_rate)
  opt_state = optimizer.init(params)
  noise_state = privatizer.init(params)

  # Batch selection setup
  batch_strategy = CyclicPoissonSampling(
      sampling_prob=batch_size / dataset_size,
      iterations=iterations,
  )
  batch_iterator = batch_strategy.batch_iterator(
      num_examples=dataset_size,
      rng=secure_rng.bit_generator,
  )

  @jax.jit
  def train_step(params, opt_state, noise_state, batch, noise):
    """Performs a single training step."""
    grads = jax.grad(loss_fn)(params, model, batch)
    noisy_grads, new_noise_state = privatizer.update(
        grads, noise_state, noise=noise
    )
    updates, new_opt_state = optimizer.update(noisy_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, new_noise_state

  # Asynchronous noise generation
  print("Generating initial secure noise...")
  future_noise = generate_secure_noise(params, stddev, secure_rng)
  future_noise = jax.device_put(future_noise)

  # Training loop
  total_time = 0
  for step_num, batch_indices in enumerate(batch_iterator):
    print(f"Step {step_num + 1}/{iterations}")
    batch = jax.tree.map(lambda x: x[batch_indices], dummy_dataset)

    # Use the noise for the current step
    current_noise = future_noise

    start_time = time.time()

    # Generate noise for the *next* step while the current step runs
    future_noise = generate_secure_noise(params, stddev, secure_rng)
    future_noise = jax.device_put(future_noise)
    
    # Perform the training step
    params, opt_state, noise_state = train_step(
        params, opt_state, noise_state, batch, current_noise
    )

    # Block to ensure the step is finished before measuring time
    jax.block_until_ready(params)
    end_time = time.time()

    step_time = end_time - start_time
    total_time += step_time
    print(f"  Step time: {step_time:.4f}s")
    print("  Secure noise injected.")

  print("\nTraining finished.")
  print(f"Average time per step: {total_time / iterations:.4f}s")


if __name__ == "__main__":
  main()
