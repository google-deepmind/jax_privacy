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
from jax_privacy import noise_addition
from numpy.random import Generator
from randomgen import AESCounter

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
def generate_secure_noise(stddev, params, generators):
  """Generates i.i.d. Gaussian noise on the CPU using randomgen."""
  return jax.tree_util.tree_map(
      lambda p, g: g.standard_normal(size=p.shape, dtype=p.dtype) * stddev,
      params,
      generators,
  )

def main(_):
  params = toy_model_params()
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

    if secure_noise is not None:
      noisy_grads = jax.tree_util.tree_map(jnp.add, grads, secure_noise)
      # We still need to update the privatizer state to keep the PRNG key state
      # consistent, even though we are not using its output.
      _, new_privatizer_state = privatizer.update(grads, privatizer_state)
    else:
      noisy_grads, new_privatizer_state = privatizer.update(
          grads, privatizer_state
      )
    return noisy_grads, new_privatizer_state

  print(f"Running {_STEPS.value} steps with use_secure_rng={_USE_SECURE_RNG.value}")
  start_time = time.time()

  # Dummy batch
  batch = None
  
  # Create a pytree of generators, one for each parameter.
  keys = jax.tree.map(
      lambda p: np.random.randint(2**63, size=2, dtype=np.uint64),
      params
  )
  generators = jax.tree.map(lambda k: Generator(AESCounter(k)), keys)

  for step in range(_STEPS.value):
    secure_noise_tree = None
    if _USE_SECURE_RNG.value:
      secure_noise_tree = generate_secure_noise(
          _STDDEV.value, params, generators
      )

    _, privatizer_state = train_step(
        params, batch, privatizer_state, secure_noise_tree
    )

  jax.block_until_ready(privatizer_state)
  end_time = time.time()

  total_time = end_time - start_time
  avg_step_time = total_time / _STEPS.value

  print(f"Total time for {_STEPS.value} steps: {total_time:.4f} seconds")
  print(f"Average Step Time: {avg_step_time:.4f} seconds")

if __name__ == '__main__':
  app.run(main)