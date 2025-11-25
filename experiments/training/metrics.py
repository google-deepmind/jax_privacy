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

""""Metrics utils."""

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp


class Avg:
  """Simple class to iteratively compute a weighted average."""

  def __init__(self):
    self._avg = 0.0
    self._n = 0

  def update(self, val, n: int = 1):
    self._avg = (self._avg * self._n + val * n) / (self._n + n)
    self._n += n

  @property
  def avg(self):
    return self._avg


def _logits_are_valid_per_row(logits: chex.Array) -> chex.Array:
  """Check per-row that no entry is set to NaN.

  Args:
    logits: array of expected shape [NO]
  Returns:
    Boolean indicator of shape [N] with the k-th entry set to True if the k-th
      row of logits contains a NaN entry, and False otherwise.
  """
  return jnp.logical_not(jnp.any(jnp.isnan(logits), axis=1))


def _labels_one_hot_are_valid_per_row(labels_one_hot: chex.Array) -> chex.Array:
  """Check per-row whether each row is a valid one-hot encoding.

  Args:
    labels_one_hot: array of expected shape [NO]
  Returns:
    Boolean indicator of shape [N] with the k-th entry set to True if the k-th
      row of `labels_one_hot` is a valid encoding, and False otherwise.
  """
  zero_or_one = jnp.all(
      labels_one_hot * labels_one_hot == labels_one_hot, axis=1)
  sum_to_one = jnp.sum(labels_one_hot, axis=1) == 1
  return jnp.logical_and(zero_or_one, sum_to_one)


def topk_accuracy(
    logits: chex.Array,
    labels_one_hot: chex.Array,
    topk: Sequence[int] = (1, 5),
) -> Sequence[jax.Array]:
  """Calculate (fast!) top-k error for multiple k values.

  Args:
    logits: array of expected shape [NO]
    labels_one_hot: array of expected shape [NO]
    topk: all k values for which top-k accuracy should be computed.
  Returns:
    Top-k accuracies for the given logits and labels.
  """
  assert logits.shape == labels_one_hot.shape

  label_scores = jnp.sum(logits * labels_one_hot, 1)

  # Compute classes that are scored at least as high as the ground truth.
  high_score_matrix = logits >= label_scores[:, jnp.newaxis]
  num_high_scores_per_sample = jnp.sum(high_score_matrix, axis=1)

  num_high_scores_per_sample = jnp.where(
      jnp.logical_and(_logits_are_valid_per_row(logits),
                      _labels_one_hot_are_valid_per_row(labels_one_hot)),
      num_high_scores_per_sample,
      jnp.inf,
  )

  # Each sample is correct for top-k accuracy if it has <= k high scores.
  return [jnp.mean(num_high_scores_per_sample <= k) for k in topk]
