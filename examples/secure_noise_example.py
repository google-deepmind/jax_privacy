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

r"""Example of how to use cryptographically secure random noise generation.

This example demonstrates how to inject custom noise into the gradient noising
process, which is useful for sourcing randomness from outside of JAX's PRNG
framework (e.g., from a hardware security module).
"""

import time
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy import noise_addition

_USE_SECURE_RNG = flags.DEFINE_boolean(
    'use_secure_rng', True, 'Whether to use secure random number generation.'
)
_STEPS = flags.DEFINE_integer(
    'steps', 10, 'Number of training steps to run.'
)
_STDDEV = flags.DEFINE_float('stddev', 1.0, 'Noise standard deviation.')

def toy_model_params():
  """Returns a PyTree of model parameters for a toy model."""
  return {
      'layer1': jnp.zeros((1024, 1024)),
      'layer2': jnp.zeros((1024, 512)),
  }

def loss_fn(params, batch):
  """A dummy loss function."""
  return sum(jnp.sum(p) for p in jax.tree.leaves(params))

# WARNING: This function must never be called inside a @jax.jit context.
# Doing so would cause the "random" noise to be statically compiled into the
# XLA graph, resulting in the same noise being added at every step.
def generate_secure_noise(stddev, grads_treedef):
  """Generates i.i.d. Gaussian noise on the CPU using NumPy."""
  return jax.tree.map(
      lambda x: np.random.normal(scale=stddev, size=x.shape).astype(x.dtype),
      grads_treedef
  )

def main(_):
  params = toy_model_params()
  # This privatizer will be used to generate noise if use_secure_rng is False
  privatizer = noise_addition.gaussian_privatizer(
      stddev=_STDDEV.value,
      prng_key=jax.random.key(0),
  )
  privatizer_state = privatizer.init(params)

  @jax.jit
  def train_step(params, batch, privatizer_state, secure_noise):
    """
    Computes gradients and adds noise.
    If `secure_noise` is provided, it's used for noising. Otherwise, the
    privatizer generates the noise.
    """
    grads = jax.grad(loss_fn)(params, batch)

    # privatizer.update still gets called to advance the PRNG state,
    # but its output is conditionally overwritten.
    noisy_grads_jax, new_privatizer_state = privatizer.update(
        grads, privatizer_state
    )

    if secure_noise is not None:
      # Manually add the CPU-generated secure noise
      iid_normal = secure_noise
      noisy_grads = jax.tree.map(
          lambda g, n: g + n, grads, iid_normal
      )
    else:
      noisy_grads = noisy_grads_jax

    return noisy_grads, new_privatizer_state

  print(f"Running {_STEPS.value} steps with use_secure_rng={_USE_SECURE_RNG.value}")
  start_time = time.time()

  # Dummy batch
  batch = None
  # We need to define grads_treedef once outside the loop
  grads_treedef = jax.eval_shape(lambda p: jax.grad(loss_fn)(p, batch), params)

  for step in range(_STEPS.value):
    secure_noise_tree = None
    if _USE_SECURE_RNG.value:
      # In a real scenario, this is where you would call your secure RNG
      secure_noise_tree = generate_secure_noise(_STDDEV.value, grads_treedef)

    # Pass the secure noise to train_step
    _, privatizer_state = train_step(
        params, batch, privatizer_state, secure_noise_tree
    )

  # Block until all steps are complete to get accurate timing
  jax.block_until_ready(privatizer_state)
  end_time = time.time()

  total_time = end_time - start_time
  avg_step_time = total_time / _STEPS.value

  print(f"Total time for {_STEPS.value} steps: {total_time:.4f} seconds")
  print(f"Average Step Time: {avg_step_time:.4f} seconds")

if __name__ == '__main__':
  app.run(main)