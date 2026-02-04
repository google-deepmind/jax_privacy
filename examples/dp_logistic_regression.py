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

"""Trains a logistic regression model with DP-BandMF."""

from absl import app
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import batch_selection
from jax_privacy.experimental import execution_plan
import numpy as np
import optax


USERS = 100_000
FEATURES = 10
EPSILON = 1.0
DELTA = 1e-5
BANDS = 4
EXPECTED_BATCH_SIZE = 1000
ITERATIONS = 500
LEARNING_RATE = 0.5
L2_CLIP_NORM = 1.0
# Trades-off recompilation with padding for variable batch sizes.
# PADDING_MULTIPLE = 1 recompiles update_fn for every batch size encountered.
# PADDING_MULTIPLE > 1 pads batches in range (kP, (k+1)P ] to size (k+1)P
PADDING_MULTIPLE = 32


def logistic_loss(params, feature_matrix, labels):
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  y_pred = 1 / (1 + jnp.exp(-logits))
  y_pred = jnp.clip(y_pred, a_min=1e-6, a_max=1 - 1e-6)
  return -jnp.mean(labels * jnp.log(y_pred) + (1 - labels) * jnp.log1p(-y_pred))


def create_benchmark(samples: int, features: int, seed: int = 0):
  """Creates a simple logistic regression model and training data."""
  key = jax.random.key(seed)
  data_key, params_key = jax.random.split(key)

  params = {
      'weights': jax.random.normal(params_key, (features,)),
      'bias': jnp.array(0.0),
  }

  feature_matrix = jax.random.normal(data_key, (samples, features))

  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  probas = 1 / (1 + jnp.exp(-logits))
  labels = np.random.rand(samples) < probas

  return params, feature_matrix, labels


def main(_):

  true_params, feature_matrix, labels = create_benchmark(USERS, FEATURES)
  params = jax.tree.map(jnp.zeros_like, true_params)
  print('Initial Loss: ', logistic_loss(params, feature_matrix, labels))

  config = execution_plan.BandMFExecutionPlanConfig(
      iterations=ITERATIONS,
      num_bands=BANDS,
      epsilon=EPSILON,
      delta=DELTA,
      sampling_prob=EXPECTED_BATCH_SIZE / USERS * BANDS,
  )
  grad_fn = jax_privacy.clipped_grad(
      logistic_loss,
      l2_clip_norm=L2_CLIP_NORM,
      batch_argnums=(1, 2),
      normalize_by=EXPECTED_BATCH_SIZE,
  )
  plan = config.make(grad_fn)

  optimizer = optax.sgd(LEARNING_RATE)
  privatizer = plan.noise_addition_transform

  @jax.jit
  def update_fn(params, batch, is_padding_example, noise_state, opt_state):
    x, y = batch
    clipped_grad = grad_fn(params, x, y, is_padding_example=is_padding_example)

    noisy_grad, noise_state = privatizer.update(clipped_grad, noise_state)
    updates, opt_state = optimizer.update(noisy_grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, noise_state, opt_state

  noise_state = privatizer.init(params)
  opt_state = optimizer.init(params)

  for batch_idx in plan.batch_selection_strategy.batch_iterator(USERS):

    # Padding reduces the required number of compilations of update_fn.
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    is_padding_example = idx == -1
    batch = feature_matrix[idx], labels[idx]

    params, noise_state, opt_state = update_fn(
        params, batch, is_padding_example, noise_state, opt_state
    )

  # loss ~ 0.27 with default parameters.
  print('Final Loss: ', logistic_loss(params, feature_matrix, labels))
  print('True Loss: ', logistic_loss(true_params, feature_matrix, labels))

  print('Learned parameters: ', params)
  print('True parameters: ', true_params)


if __name__ == '__main__':
  app.run(main)
