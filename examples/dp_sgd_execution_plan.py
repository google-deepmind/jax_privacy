# coding=utf-8
# Copyright 2026 DeepMind Technologies Limited.
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

"""Trains logistic regression with Poisson DP-SGD via DpsgdConfig.

Demonstrates the Tier-2 execution-plan path for vanilla DP-SGD: calibration,
Poisson batch selection, clipped gradients normalized by expected batch size,
and i.i.d. Gaussian noise whose scale matches the plan's dp_event.
"""

from typing import Any, Mapping, Tuple

from absl import app
import jax
import jax.numpy as jnp
from jax_privacy import batch_selection
from jax_privacy import execution_plan
import numpy as np
import optax

USERS = 5_000
FEATURES = 50
EPSILON = 2.0
DELTA = 1e-5
EXPECTED_BATCH_SIZE = 250
ITERATIONS = 200
LEARNING_RATE = 0.25
L2_CLIP_NORM = 1.0
PADDING_MULTIPLE = 32


def logistic_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Per-example mean logistic loss (batch size 1 under clipped_grad)."""
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  signed_logits = jnp.where(labels, logits, -logits)
  return -jnp.mean(jax.nn.log_sigmoid(signed_logits))


def create_benchmark(
    samples: int,
    features: int,
    seed: int = 0,
) -> Tuple[Mapping[str, jax.Array], jax.Array, jax.Array]:
  """Creates a logistic regression model and synthetic training data."""
  key = jax.random.key(seed)
  data_key, params_key = jax.random.split(key)
  params = {
      'weights': jax.random.normal(params_key, (features,)),
      'bias': jnp.array(0.0),
  }
  feature_matrix = jax.random.normal(data_key, (samples, features))
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  labels = np.random.default_rng(seed).random(samples) < jax.nn.sigmoid(logits)
  return params, feature_matrix, labels  # pyrefly: ignore[bad-return]


def main(_):
  true_params, features, labels = create_benchmark(USERS, FEATURES)
  init_params = jax.tree.map(jnp.zeros_like, true_params)
  init_loss = logistic_loss(init_params, features, labels)
  print(f'Initial training loss: {init_loss:.3f}')

  config = execution_plan.DpsgdConfig(
      iterations=ITERATIONS,
      expected_batch_size=EXPECTED_BATCH_SIZE,
      num_examples=USERS,
      l2_clip_norm=L2_CLIP_NORM,
      rescale_to_unit_norm=True,
  ).calibrate(epsilon=EPSILON, delta=DELTA)
  print(
      'Calibrated DpsgdConfig with'
      f' noise_multiplier={config.noise_multiplier:.4f}'
  )
  plan = config.make(execution_plan.PerformanceFlags(noise_seed=0))
  grad_fn = plan.clipped_grad(logistic_loss, batch_argnums=(1, 2))
  optimizer = optax.sgd(LEARNING_RATE)

  @jax.jit
  def update_fn(
      params: Mapping[str, jax.Array],
      batch: Tuple[jax.Array, jax.Array],
      is_padding_example: jax.Array,
      noise_state: Any,
      opt_state: optax.OptState,
  ) -> Tuple[Mapping[str, jax.Array], Any, optax.OptState]:
    x, y = batch
    clipped_grad_sum = grad_fn(
        params, x, y, is_padding_example=is_padding_example
    )
    noisy_grad, noise_state = plan.noise_addition_transform.update(
        clipped_grad_sum, noise_state
    )
    updates, opt_state = optimizer.update(noisy_grad, opt_state)
    # pyrefly: ignore[bad-assignment]
    params = optax.apply_updates(params, updates)
    return params, noise_state, opt_state

  params = init_params
  noise_state = plan.noise_addition_transform.init(params)
  opt_state = optimizer.init(params)

  for step, batch_idx in enumerate(
      plan.batch_selection_strategy.batch_iterator(USERS, rng=0)
  ):
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    is_padding_example = idx == -1
    batch = features[idx], labels[idx]
    params, noise_state, opt_state = update_fn(
        params, batch, is_padding_example, noise_state, opt_state
    )
    if step > 0 and step % 50 == 0:
      loss = logistic_loss(params, features, labels)
      print(f'Step {step:4d}, loss={loss:.4f}')

  final_loss = float(logistic_loss(params, features, labels))
  print(f'Final training loss: {final_loss:.4f}')
  assert final_loss < float(init_loss), 'DP-SGD failed to reduce training loss.'
  print('DpsgdConfig training completed successfully.')


if __name__ == '__main__':
  app.run(main)
