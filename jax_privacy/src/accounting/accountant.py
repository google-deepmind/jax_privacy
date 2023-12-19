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

from collections.abc import Sequence
import time
from typing import Protocol, TypeAlias

from absl import logging
import chex
from jax_privacy.src.accounting import calibrate
from jax_privacy.src.accounting import dp_bounds
import numpy as np
import optax


def _to_python_int(value_or_array: chex.Numeric) -> int:
  """Converts the input to a python int (fails if value is not unique)."""
  # Expecting a unique value even if the array can have multiple entries.
  [single_value] = np.unique(value_or_array)
  return int(single_value)


# The batch-size scale schedule allows to change the schedule during training:
# e.g. if `bs_init = 16` and `bs_schedule = {100: 4, 200: 2}`, then the
# batch-size is set to 16 on steps 0-99, to 16*4 on steps 100-199, and
# to 16*2 from step 200 onwards.
BatchingScaleSchedule: TypeAlias = dict[int, int] | None


def _make_batch_size_boundaries(
    batch_size: int,
    scale_schedule: BatchingScaleSchedule,
) -> Sequence[tuple[int, chex.Numeric]]:
  """Returns the boundaries for batch-size values."""
  schedule_fn = optax.piecewise_constant_schedule(
      init_value=batch_size,
      boundaries_and_scales=scale_schedule,
  )
  return [(0, batch_size)] + [
      (threshold, schedule_fn(threshold + 1))
      for threshold in scale_schedule
  ]


class Accountant(Protocol):

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Computes epsilon given the DP parameters and current `num_updates`."""


class ExperimentAccountant:
  """Tracks privacy spent and maximal number of updates for an experiment."""

  def __init__(
      self,
      noise_multiplier: float | None,
      dp_delta: float,
      num_samples: int,
      dp_accountant_config: dp_bounds.DpAccountantConfig,
      batch_size: int,
      batch_size_scale_schedule: BatchingScaleSchedule = None,
      is_finite_guarantee: bool = True,
  ):
    """Initializes the accountant for Differential Privacy.

    Args:
      noise_multiplier: The noise multiplier, excluding the clipping norm and
        the batch-size.
      dp_delta: delta-value of DP guarantee.
      num_samples: number of examples in the training set.
      dp_accountant_config: Configuration for the DP accountant to use.
      batch_size: batch-size used during training.
      batch_size_scale_schedule: schedule for scaling the batch-size.
      is_finite_guarantee: Whether the DP guarantee can be expected to be
        finite. This may be False if the clipping norm is not finite for
        example.
    """
    self._noise_multiplier = noise_multiplier
    self._num_samples = num_samples
    self._is_finite_guarantee = is_finite_guarantee
    self._dp_delta = dp_delta

    if batch_size_scale_schedule:
      self._batch_sizes = _make_batch_size_boundaries(
          batch_size, batch_size_scale_schedule)
    else:
      self._batch_sizes = batch_size

    self._dp_accountant_config = dp_accountant_config

  def compute_max_num_updates(self, epsilon: float) -> int:
    """Compute maximum number of updates given the DP parameters."""
    if self._is_finite_guarantee:
      return calibrate.calibrate_steps(
          target_epsilon=epsilon,
          noise_multipliers=self._noise_multiplier,
          batch_sizes=self._batch_sizes,
          num_examples=self._num_samples,
          target_delta=self._dp_delta,
          dp_accountant_config=self._dp_accountant_config,
      )
    else:
      return 0

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Compute DP epsilon given the DP parameters and current `num_updates`."""
    del allow_approximate_cache  # this class never uses an approximate cache.
    num_updates = _to_python_int(num_updates)
    if num_updates == 0:
      return 0.0
    elif self._is_finite_guarantee:
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
    self._cache_is_initialized = False

  def _maybe_initialize_cache(self):
    """Precomputes and caches the values of `num_cached_points` points."""
    if self._cache_is_initialized:
      return
    logging.info('Pre-computing accounting cache...')
    start_clock = time.time()
    self._cached_points = [
        _ceil_div(self._max_num_updates * j, self._num_cached_points)
        for j in range(self._num_cached_points + 1)
    ]
    self._cached_values = {
        x: self._accountant.compute_epsilon(x)
        for x in self._cached_points}
    end_clock = time.time()
    logging.info(
        'Accounting cache (took %.3f).', end_clock - start_clock)
    self._cache_is_initialized = True

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Uses cached results to give an approximate (over-estimated) epsilon.

    The value returned should always be an over-approximation of the true
    epsilon: this method uses the closest `num_updates` in the cache that is
    equal to or greater than the requested `num_updates`. If such a value cannot
    be found, an indexing error will be raised.

    Args:
      num_updates: Number of updates for which to compute epsilon.
      allow_approximate_cache: Whether to use the approximate cache.
    Returns:
      Value of epsilon.
    """
    if allow_approximate_cache:
      self._maybe_initialize_cache()
      num_updates = _to_python_int(num_updates)
      closest_cached_point = self._cached_points[
          _ceil_div(
              self._num_cached_points * num_updates, self._max_num_updates
          )
      ]
      return self._cached_values[closest_cached_point]
    else:
      return self._accountant.compute_epsilon(num_updates)
