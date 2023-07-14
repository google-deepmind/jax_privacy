# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

"""Keeping track of the differential privacy guarantee."""

from typing import Optional

from jax_privacy.src.accounting import calibrate
from jax_privacy.src.accounting import dp_bounds
from jax_privacy.src.dp_sgd import batching as batching_py
import numpy as np


class ExperimentAccountant:
  """Tracks privacy spent and maximal number of updates for an experiment."""

  def __init__(
      self,
      clipping_norm: Optional[float],
      noise_multiplier: Optional[float],
      dp_epsilon: float,
      dp_delta: float,
      num_samples: int,
      batching: batching_py.VirtualBatching,
      dp_accountant_config: dp_bounds.DpAccountantConfig,
  ):
    """Initializes the accountant for Differential Privacy.

    This class wraps the functions defined in `calibrate.py` and `dp_bounds.py`
    so that we can gracefully handle limit cases where either
    `noise_multiplier=0` or the clipping norm is infinite, which both result in
    infinite (vacuous) DP guarantees.

    Args:
      clipping_norm: clipping-norm for the per-example gradients (before
        averaging across the examples of the mini-batch).
      noise_multiplier: The noise multiplier, excluding the clipping norm and
        the batch-size.
      dp_epsilon: epsilon-value of DP guarantee.
      dp_delta: delta-value of DP guarantee.
      num_samples: number of examples in the training set.
      batching: batch-size used during training.
      dp_accountant_config: Configuration for the DP accountant to use.
    """
    if clipping_norm is None:
      self._clipping_norm = float('inf')
    elif clipping_norm < 0:
      raise ValueError('Clipping norm must be non-negative.')
    else:
      self._clipping_norm = clipping_norm

    if noise_multiplier is None:
      self._noise_multiplier = 0
    elif noise_multiplier < 0:
      raise ValueError('Standard deviation must be non-negative.')
    else:
      self._noise_multiplier = noise_multiplier

    self._batching = batching
    self._num_samples = num_samples

    self._dp_epsilon = dp_epsilon
    self._dp_delta = dp_delta

    self._batch_sizes = [(0, self._batching.batch_size_init)]
    if self._batching.scale_schedule is not None:
      self._batch_sizes.extend(
          [(threshold, self._batching.batch_size(threshold+1))
           for threshold in self._batching.scale_schedule]
      )

    self._dp_accountant_config = dp_accountant_config

  def finite_dp_guarantee(self) -> bool:
    """Returns whether the DP guarantee (eps, delta) can be finite."""
    # The privacy (eps, delta) can only be finite with non-zero noise
    # and with a finite clipping-norm.
    return bool(self._noise_multiplier and np.isfinite(self._clipping_norm))

  def compute_max_num_updates(self) -> int:
    """Compute maximum number of updates given the DP parameters."""
    if self.finite_dp_guarantee():
      return calibrate.calibrate_steps(
          target_epsilon=self._dp_epsilon,
          noise_multipliers=self._noise_multiplier,
          batch_sizes=self._batch_sizes,
          num_examples=self._num_samples,
          target_delta=self._dp_delta,
          dp_accountant_config=self._dp_accountant_config,
      )
    else:
      return 0

  def compute_current_epsilon(self, num_updates: int) -> float:
    """Compute DP epsilon given the DP parameters and current `num_updates`."""
    if num_updates == 0:
      return 0.0
    elif self.finite_dp_guarantee():
      return dp_bounds.compute_epsilon(
          num_steps=num_updates,
          noise_multipliers=self._noise_multiplier,
          batch_sizes=self._batch_sizes,
          num_examples=self._num_samples,
          target_delta=self._dp_delta,
          dp_accountant_config=self._dp_accountant_config,
      )
    else:
      return float('inf')


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


class CachedExperimentAccountant:
  """Pre-computes and caches epsilon for different `num_updates` values."""

  def __init__(
      self,
      accountant: ExperimentAccountant,
      max_num_updates: int,
      num_cached_points: int = 100,
  ):
    """Creates the cached accoutant and computes the cached points and values.

    Args:
      accountant: DP accountant to use for computing the results to be cached.
      max_num_updates: Maximum value for `num_updates` to be requested.
      num_cached_points: Number of points to pre-compute and cache.
    """
    self._accountant = accountant
    self._max_num_updates = max_num_updates
    self._num_cached_points = num_cached_points

    self._cached_points = [
        _ceil_div(self._max_num_updates * j, self._num_cached_points)
        for j in range(self._num_cached_points + 1)
    ]
    self._cached_values = {
        x: self._accountant.compute_current_epsilon(x)
        for x in self._cached_points}

  def compute_approximate_epsilon(self, num_updates: int) -> float:
    """Uses cached results to give an approximate (over-estimated) epsilon.

    The value returned should always be an over-approximation of the true
    epsilon: this method uses the closest `num_updates` in the cache that is
    equal to or greater than the requested `num_updates`. If such a value cannot
    be found, an indexing error will be raised.

    Args:
      num_updates: Number of updates for which to compute epsilon.
    Returns:
      Approximate (over-estimated) value of epsilon.
    """
    closest_cached_point = self._cached_points[
        _ceil_div(self._num_cached_points * num_updates, self._max_num_updates)]
    return self._cached_values[closest_cached_point]
