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

"""Trains a logistic regression model with DP-BandMF."""

from typing import Any, Mapping, Tuple

from absl import app
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import auditing
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
TRAIN_SPLIT_PERCENT = 0.9
# Trades-off recompilation with padding for variable batch sizes.
# PADDING_MULTIPLE = 1 recompiles update_fn for every batch size encountered.
# PADDING_MULTIPLE > 1 pads batches in range (kP, (k+1)P ] to size (k+1)P
PADDING_MULTIPLE = 32


def elementwise_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Computes element-wise loss for auditing."""
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  log_p = jax.nn.log_sigmoid(logits)
  log_one_minus_p = jax.nn.log_sigmoid(-logits)
  return -(labels * log_p + (1 - labels) * log_one_minus_p)


def logistic_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  return jnp.mean(elementwise_loss(params, feature_matrix, labels))


def create_benchmark(
    samples: int,
    features: int,
    seed: int = 0,
) -> Tuple[Mapping[str, jax.Array], jax.Array, jax.Array]:
  """Creates a simple logistic regression model and training data."""
  key = jax.random.key(seed)
  data_key, params_key = jax.random.split(key)

  params = {
      'weights': jax.random.normal(params_key, (features,)),
      'bias': jnp.array(0.0),
  }

  feature_matrix = jax.random.normal(data_key, (samples, features))

  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  probas = jax.nn.sigmoid(logits)
  labels = np.random.rand(samples) < probas

  return params, feature_matrix, labels


def main(_):
  # Split data into training set (in-canaries) and audit set (out-canaries)
  total_users = USERS
  train_users = int(total_users * TRAIN_SPLIT_PERCENT)
  true_params, all_features, all_labels = create_benchmark(
      total_users, FEATURES
  )

  # Split dataset
  train_features, train_labels = (
      all_features[:train_users],
      all_labels[:train_users],
  )
  audit_features, audit_labels = (
      all_features[train_users:],
      all_labels[train_users:],
  )

  params = jax.tree.map(jnp.zeros_like, true_params)
  init_params = params
  print(
      'Initial Loss(Train): ',
      logistic_loss(params, train_features, train_labels),
  )

  config = execution_plan.BandMFExecutionPlanConfig(
      iterations=ITERATIONS,
      num_bands=BANDS,
      epsilon=EPSILON,
      delta=DELTA,
      sampling_prob=EXPECTED_BATCH_SIZE / train_users * BANDS,
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
  def update_fn(
      params: Mapping[str, jax.Array],
      batch: Tuple[jax.Array, jax.Array],
      is_padding_example: jax.Array,
      noise_state: Any,
      opt_state: optax.OptState,
  ) -> Tuple[Mapping[str, jax.Array], Any, optax.OptState]:
    x, y = batch
    clipped_grad = grad_fn(params, x, y, is_padding_example=is_padding_example)

    noisy_grad, noise_state = privatizer.update(clipped_grad, noise_state)
    updates, opt_state = optimizer.update(noisy_grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, noise_state, opt_state

  noise_state = privatizer.init(params)
  opt_state = optimizer.init(params)

  for batch_idx in plan.batch_selection_strategy.batch_iterator(train_users):

    # Padding reduces the required number of compilations of update_fn.
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    is_padding_example = idx == -1
    batch = train_features[idx], train_labels[idx]

    params, noise_state, opt_state = update_fn(
        params, batch, is_padding_example, noise_state, opt_state
    )

  # loss ~ 0.27 with default parameters.
  print(
      'Final Loss(Train): ', logistic_loss(params, train_features, train_labels)
  )
  print(
      'True Loss(Train): ',
      logistic_loss(true_params, train_features, train_labels),
  )

  print('Learned parameters: ', params)
  print('True parameters: ', true_params)

  # --- Canary Insertion / Auditing ---
  print('\n--- Privacy Auditing ---')

  # Calculate scores (Initial Loss - Final Loss)
  in_init_loss = elementwise_loss(init_params, train_features, train_labels)
  in_final_loss = elementwise_loss(params, train_features, train_labels)
  in_scores = in_init_loss - in_final_loss

  out_init_loss = elementwise_loss(init_params, audit_features, audit_labels)
  out_final_loss = elementwise_loss(params, audit_features, audit_labels)
  out_scores = out_init_loss - out_final_loss

  # Subsample in_scores to match out_scores size
  in_scores = in_scores[: len(out_scores)]

  # Run Auditor
  auditor = auditing.CanaryScoreAuditor(in_scores, out_scores)

  # Estimate epsilon
  estimated_epsilon = auditor.epsilon_raw_counts(min_count=10, delta=DELTA)

  # 99% upper confidence interval
  quantile_99 = auditing.BootstrapParams(quantiles=0.99)
  estimated_epsilon_99 = auditor.epsilon_raw_counts(
      min_count=10, delta=DELTA, bootstrap_params=quantile_99
  )
  auroc_99 = auditor.attack_auroc(bootstrap_params=quantile_99)

  print(f'Theoretical Epsilon: {EPSILON:.2f}')
  print(f'Estimated Epsilon:   {estimated_epsilon:.2f}')
  print(f'Estimated Epsilon (99% CI): {estimated_epsilon_99:.2f}')
  print(f'AUROC (99% CI):      {auroc_99:.4f}')
  print(
      f'Start Auditing with {len(in_scores)} in-canaries and {len(out_scores)}'
      ' out-canaries.'
  )


if __name__ == '__main__':
  app.run(main)
