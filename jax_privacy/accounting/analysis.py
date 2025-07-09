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

"""Keeping track of the differential privacy guarantee."""

import abc
from collections.abc import Sequence
import dataclasses
import enum
import math
import numbers
import time
from typing import Protocol, TypeAlias

from absl import logging
import chex
import dp_accounting
from jax_privacy.accounting import accountants
import numpy as np
import optax
from scipy import stats


class SamplingMethod(enum.Enum):
  """The sampling method assumed by the privacy analysis.

  `POISSON`:
    We assume each element is independently included in the batch with
    probability `batch_size / num_samples`, such that the batch has expected
    size `batch_size`. Compatible with add-or-remove and zero-out adjacencies.
  `FIXED_BATCH_SIZE`:
    We assume that the batch is a random subset of size `batch_size`. Compatible
    with replace adjacency (and thus zero-out), assuming that `num_samples` is
    public knowledge. The reported DP guarantee is also valid for add-or-remove
    adjacency, but replace adjacency is considered stronger so this is not
    recommended.
  """
  POISSON = enum.auto()
  FIXED_BATCH_SIZE = enum.auto()


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


def _interleave_nm_and_bs(
    noise_multipliers: float | Sequence[tuple[int, float]],
    batch_sizes: int | Sequence[tuple[int, int]],
    num_steps: int,
) -> Sequence[tuple[int, float, int]]:
  """Returns `noise_multipliers` and `batch_sizes` across `num_steps`."""

  # If noise_multipliers is a number, turn it into list format of (0, nm).
  if isinstance(noise_multipliers, numbers.Number):
    noise_multipliers = [(0, noise_multipliers)]
  # If batch_sizes is a number, turn it into list format of (0, bs).
  if isinstance(batch_sizes, int):
    batch_sizes = [(0, batch_sizes)]

  # Make sure the time steps of changes are increasing.
  noise_multipliers = sorted(noise_multipliers, key=lambda t: t[0])
  batch_sizes = sorted(batch_sizes, key=lambda x: x[0])
  # Make sure the first time step is 0 in both sequences of hyper-parameters.
  assert noise_multipliers[0][0] == 0
  assert batch_sizes[0][0] == 0
  # Remove any settings which occur later than the maximum number of steps.
  noise_multipliers = [(t, x) for t, x in noise_multipliers if t <= num_steps]
  batch_sizes = [x for x in batch_sizes if x[0] <= num_steps]

  # Interleave both sequences of hyper-parameters into a single one.
  nm_and_bs = _interleave(noise_multipliers, batch_sizes)
  t_nm_and_bs = []
  # Adjust time indices to count number of steps in each configuration.
  for i in range(len(nm_and_bs) - 1):
    t_nm_and_bs.append((
        nm_and_bs[i + 1][0] - nm_and_bs[i][0],
        nm_and_bs[i][1],
        nm_and_bs[i][2],
    ))
  t_nm_and_bs.append(
      (num_steps - nm_and_bs[-1][0], nm_and_bs[-1][1], nm_and_bs[-1][2])
  )
  return t_nm_and_bs


def _interleave(t_a, t_b):
  """Helper function to pair two timed sequences."""
  ts = [t for (t, _) in t_a] + [t for (t, _) in t_b]
  ts = list(set(ts))
  ts.sort()

  def _find_pair(t):
    a = [a for (s, a) in t_a if s <= t][-1]
    b = [b for (s, b) in t_b if s <= t][-1]
    return a, b

  return [(t, *_find_pair(t)) for t in ts]


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
      (threshold, schedule_fn(threshold + 1)) for threshold in scale_schedule
  ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class DpParams:
  """Defines static parameters required for computing DP guarantees.

  Attributes:
    noise_multipliers: The noise multiplier, excluding the clipping norm and the
      batch-size. Or, list of pairs (t: int, nm: float) if the noise multiplier
      changes across steps. 't' indicates step where noise_multiplier is set to
      'nm'.
    delta: delta-value of DP guarantee.
    num_samples: number of examples in the training set.
    batch_size: batch-size used during training.
    dp_analysis_algorithm_config: Configuration for the accounting analysis
      algorithm to use. See subclasses of `DpAnalysisAlgorithmConfig`.
    batch_size_scale_schedule: schedule for scaling the batch-size.
    is_finite_guarantee: Whether the DP guarantee can be expected to be finite.
      This may be False if the clipping norm is not finite for example.
    batch_sizes: (Read-only) attribute consisting of list of pairs (t: int,
      bs: int) where `t` indicates the step where `batch_size` is set to `bs`.
        This is computed from `batch_size` and `batch_size_scale_schedule`.
    examples_per_user: If multiple examples per user are used, this is the
      maximum number any user contributes to the training set.
    cycle_length: If using cyclic Poisson sampling with BandMF, the length of
      the cycle, i.e. the number of partitions formed for sampling. It is
      assumed the number of bands in BandMF is at most cycle_length.
    sampling_method: If our privacy analysis assumes sampling, which sampling
      method it should assume. See SamplingMethod enum for details on each
      sampling method and the adjacency definitions it assumes.
    truncated_batch_size: If using Poisson sampling, a limit on the batch size
      enforced by truncation. If None, we assume no truncation is used.
      Otherwise, we assume that a random subset of the sampled batch is used,
      and the remaining examples are discarded.
  """

  noise_multipliers: float | Sequence[tuple[int, float]] | None
  num_samples: int
  delta: float
  batch_size: int
  batch_size_scale_schedule: BatchingScaleSchedule = None
  is_finite_guarantee: bool = True
  batch_sizes: Sequence[tuple[int, chex.Numeric]] | int = dataclasses.field(
      init=False
  )
  examples_per_user: int | None = None
  cycle_length: int | None = None
  sampling_method: SamplingMethod = SamplingMethod.POISSON
  truncated_batch_size: int | None = None

  def __post_init__(self):
    if self.batch_size_scale_schedule:
      batch_sizes = _make_batch_size_boundaries(
          self.batch_size, self.batch_size_scale_schedule
      )
    else:
      batch_sizes = self.batch_size
    object.__setattr__(self, 'batch_sizes', batch_sizes)


class TrainingAccountant(Protocol):

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      dp_params: DpParams,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Computes epsilon given the DP parameters and current `num_updates`."""


class DpTrainingAccountant(metaclass=abc.ABCMeta):
  """Defines privacy accounting interface for machine learning training."""

  def __init__(
      self,
      dp_accountant_config: accountants.DpAccountantConfig,
  ):
    """Initializes the accountant for Differential Privacy.

    Args:
      dp_accountant_config: Configuration for the DP accountant to use.
    """
    self._dp_accountant_config = dp_accountant_config

  @abc.abstractmethod
  def _compute_epsilon(
      self, num_updates: chex.Numeric, dp_params: DpParams
  ) -> float:
    """Computes epsilon using `num_updates` and `dp_params`."""

  @abc.abstractmethod
  def can_calibrate_steps(self) -> bool:
    """Returns whether the `num_steps` can be calibrated."""

  @abc.abstractmethod
  def can_calibrate_batch_size(self) -> bool:
    """Returns whether the `batch_size` can be calibrated."""

  @abc.abstractmethod
  def can_calibrate_noise_multipliers(self) -> bool:
    """Returns whether the `noise_multipliers` can be calibrated."""

  def _validate_dp_params(self, dp_params: DpParams):
    """Asserts that the accountant supports the given `dp_params`."""
    if dp_params.noise_multipliers is None:
      raise ValueError(f'{self.__class__.__name__} requires noise_multipliers.')

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      dp_params: DpParams,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Compute DP epsilon given the `dp_params`."""
    del allow_approximate_cache  # This class never uses allow_approximate_cache

    if num_updates == 0:
      return 0.0
    elif dp_params.is_finite_guarantee:
      self._validate_dp_params(dp_params)
      return self._compute_epsilon(num_updates, dp_params)
    else:
      return float('inf')


class DpsgdTrainingAccountant(DpTrainingAccountant):
  """Defines privacy computations for Band-MF with Cyclic Poisson sampling.

  This includes DP-SGD style analysis as a special case.

  For accounting we follow the reduction in https://arxiv.org/abs/2306.08153. We
  assume that if num_samples % cycle_length != 0, then num_samples %
  cycle_length examples are discarded.
  """

  def can_calibrate_steps(self) -> bool:
    return True

  def can_calibrate_batch_size(self) -> bool:
    return True

  def can_calibrate_noise_multipliers(self) -> bool:
    return True

  def _validate_dp_params(self, dp_params: DpParams):
    super()._validate_dp_params(dp_params)
    if (
        dp_params.examples_per_user is not None
        and dp_params.examples_per_user != 1
    ):
      raise ValueError(
          'DpsgdTrainingAccountant requires examples_per_user = 1 or None. '
          'Choose a different examples_per_user or use '
          'DpsgdTrainingUserLevelAccountant instead.'
      )
    if dp_params.cycle_length is not None and dp_params.cycle_length != 1:
      if not isinstance(dp_params.batch_size, numbers.Number):
        raise ValueError(
            'DpsgdTrainingAccountant with cycle_length != 1 requires a single'
            ' batch size.'
        )
      if not isinstance(dp_params.noise_multipliers, numbers.Number):
        raise ValueError(
            'DpsgdTrainingAccountant with cycle_length != 1 requires a single'
            ' noise multiplier.'
        )
      if dp_params.batch_size * dp_params.cycle_length > dp_params.num_samples:
        raise ValueError(
            'DpsgdTrainingAccountant with cycle_length != 1 requires batch_size'
            ' * cycle_length <= num_samples.'
        )
    if (
        dp_params.sampling_method is not SamplingMethod.POISSON
        and dp_params.sampling_method is not SamplingMethod.FIXED_BATCH_SIZE
    ):
      raise ValueError(
          'DpsgdTrainingAccountant requires sampling_method = POISSON or '
          'FIXED_BATCH_SIZE.'
      )
    if dp_params.truncated_batch_size is not None:
      if dp_params.sampling_method is not SamplingMethod.POISSON:
        raise ValueError(
            'DpsgdTrainingAccountant does not support truncated_batch_size'
            ' unless using sampling_method = POISSON.'
        )
      if not isinstance(
          self._dp_accountant_config, accountants.PldAccountantConfig
      ):
        raise ValueError(
            'DpsgdTrainingAccountant with truncated_batch_size != None requires'
            ' a PLDAccountant.'
        )

  def _compute_epsilon(
      self, num_updates: chex.Numeric, dp_params: DpParams
  ) -> float:
    nms = dp_params.noise_multipliers
    batch_sizes = dp_params.batch_sizes
    num_samples = dp_params.num_samples
    sampling_method = dp_params.sampling_method
    cycle_length = dp_params.cycle_length if dp_params.cycle_length else 1
    truncated_batch_size = dp_params.truncated_batch_size
    dp_accountant = self._dp_accountant_config.create_accountant()

    t_nm_and_bs = _interleave_nm_and_bs(nms, batch_sizes, num_updates)
    match sampling_method:
      case SamplingMethod.POISSON:
        sensitivity_multiplier = 1.0
      case SamplingMethod.FIXED_BATCH_SIZE:
        # Fixed batch size sampling's privacy analysis reduces to Poisson
        # sampling with the same noise but the sensitivity doubled.
        sensitivity_multiplier = 2.0
    for t, nm, bs in t_nm_and_bs:
      min_group_size = num_samples // cycle_length
      q = bs / float(min_group_size)
      if truncated_batch_size is None:
        event = dp_accounting.PoissonSampledDpEvent(
            q, dp_accounting.GaussianDpEvent(nm / sensitivity_multiplier)
        )
      else:
        # This calculation involves a sum over num_samples terms corresponding
        # to the possible batch sizes before truncation. To save time and memory
        # we truncate this sum at a threshold chosen such that the terms in the
        # sum after the threshold are smaller than the precision of computation.
        threshold = truncated_batch_size
        while stats.binom.sf(threshold, min_group_size - 1, q) > 0.0:
          threshold = max(2 * threshold, min_group_size)
        sample_sizes = np.arange(truncated_batch_size, threshold)
        prob_2 = q * np.sum(
            stats.binom.pmf(sample_sizes, min_group_size - 1, q)
            * truncated_batch_size
            / (sample_sizes + 1)
        )
        prob_1 = q * (
            1 - stats.binom.sf(truncated_batch_size, min_group_size - 1, q)
        )
        prob_0 = 1 - prob_1 - prob_2
        event = dp_accounting.dp_event.MixtureOfGaussiansDpEvent(
            nm, [0, 1, 2], [prob_0, prob_1, prob_2]
        )
      dp_accountant.compose(event, math.ceil(t / cycle_length))
    return dp_accountant.get_epsilon(target_delta=dp_params.delta)


class DpsgdTrainingUserLevelAccountant(DpTrainingAccountant):
  """Defines privacy computations for DP-SGD analysis with user-level DP.

  This class uses the calculations in https://arxiv.org/abs/2401.10294.
  """

  def can_calibrate_steps(self) -> bool:
    return True

  def can_calibrate_batch_size(self) -> bool:
    return True

  def can_calibrate_noise_multipliers(self) -> bool:
    return True

  def _validate_dp_params(self, dp_params: DpParams):
    super()._validate_dp_params(dp_params)
    if dp_params.examples_per_user is None:
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires examples_per_user.'
      )
    if dp_params.cycle_length is not None and dp_params.cycle_length != 1:
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires cycle_length = 1 or None.'
      )
    if (
        dp_params.sampling_method is not SamplingMethod.POISSON
        and dp_params.sampling_method is not SamplingMethod.FIXED_BATCH_SIZE
    ):
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires sampling_method = POISSON '
          'or FIXED_BATCH_SIZE.'
      )
    if dp_params.truncated_batch_size is not None:
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires truncated_batch_size ='
          ' None.'
      )
    if not isinstance(
        self._dp_accountant_config, accountants.PldAccountantConfig
    ):
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires a PLDAccountant.'
      )

  def _compute_epsilon(
      self, num_updates: chex.Numeric, dp_params: DpParams
  ) -> float:
    nms = dp_params.noise_multipliers
    batch_sizes = dp_params.batch_sizes
    num_samples = dp_params.num_samples
    examples_per_user = dp_params.examples_per_user
    sampling_method = dp_params.sampling_method
    dp_accountant = self._dp_accountant_config.create_accountant()
    if not isinstance(dp_accountant, dp_accounting.pld.PLDAccountant):
      raise ValueError(
          'DpsgdTrainingUserLevelAccountant requires a PLDAccountant.'
      )

    t_nm_and_bs = _interleave_nm_and_bs(nms, batch_sizes, num_updates)
    for t, nm, bs in t_nm_and_bs:
      match sampling_method:
        case SamplingMethod.POISSON:
          q = bs / float(num_samples)
          sensitivities = range(examples_per_user + 1)
          probs = [
              stats.binom.pmf(x, examples_per_user, q) for x in sensitivities
          ]
        case SamplingMethod.FIXED_BATCH_SIZE:
          sensitivities = [2 * x for x in range(examples_per_user + 1)]
          sensitivity_rv = stats.hypergeom(num_samples, examples_per_user, bs)
          probs = [sensitivity_rv.pmf(x) for x in range(examples_per_user + 1)]
      event = dp_accounting.dp_event.MixtureOfGaussiansDpEvent(
          nm, sensitivities, probs
      )
      dp_accountant.compose(event, t)
    return dp_accountant.get_epsilon(target_delta=dp_params.delta)


class SingleReleaseTrainingAccountant(DpTrainingAccountant):
  """Defines privacy computations for single release analysis.

  This style of analysis is used for un-amplified DP-FTRL mechanisms, as
  detailed in https://arxiv.org/pdf/2211.06530. Unlike DP-SGD analysis, which
  relies on Poisson amplification, this analysis treats accounting as a single
  Gaussian DP event.
  """

  def can_calibrate_steps(self) -> bool:
    return False

  def can_calibrate_batch_size(self) -> bool:
    return False

  def can_calibrate_noise_multipliers(self) -> bool:
    return False

  def _validate_dp_params(self, dp_params: DpParams):
    super()._validate_dp_params(dp_params)
    if (
        dp_params.examples_per_user is not None
        and dp_params.examples_per_user != 1
    ):
      raise ValueError(
          'SingleReleaseTrainingAccountant requires examples_per_user = 1 or'
          ' None'
      )

  def _compute_epsilon(
      self, num_updates: chex.Numeric, dp_params: DpParams
  ) -> float:
    nms = dp_params.noise_multipliers
    batch_sizes = dp_params.batch_sizes
    dp_accountant = self._dp_accountant_config.create_accountant()

    t_nm_and_bs = _interleave_nm_and_bs(nms, batch_sizes, num_updates)
    for _, nm, _ in t_nm_and_bs:
      event = dp_accounting.GaussianDpEvent(nm)
      dp_accountant.compose(event, 1)
    return dp_accountant.get_epsilon(target_delta=dp_params.delta)


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


class CachedExperimentAccountant:
  """Pre-computes and caches epsilon for different `num_updates` values."""

  def __init__(
      self,
      training_accountant: DpTrainingAccountant,
      max_num_updates: int,
      num_cached_points: int = 100,
  ):
    """Creates the cached accoutant and computes the cached points and values.

    Args:
      training_accountant: Which training accountant to use for computing the
        results to be cached.
      max_num_updates: Maximum value for `num_updates` to be requested.
      num_cached_points: Number of points to pre-compute and cache.
    """
    self._accountant = training_accountant
    self._max_num_updates = max_num_updates
    self._num_cached_points = num_cached_points
    self._cache_is_initialized = False

  def _maybe_initialize_cache(self, dp_params: DpParams):
    """Precomputes and caches the values of `num_cached_points` points."""
    if self._cache_is_initialized:
      return
    logging.info('Pre-computing accounting cache...')
    start_clock = time.time()
    self._cached_points = [
        _ceil_div(self._max_num_updates * j, self._num_cached_points)
        for j in range(self._num_cached_points + 1)
    ]

    self._cached_values = {}
    for i, x in enumerate(self._cached_points):
      self._cached_values[x] = self._accountant.compute_epsilon(x, dp_params)

      # Compute current duration in seconds.
      current_duration = time.time() - start_clock
      ten_minutes_threshold = 10 * 60

      # Estimate the total duration by extrapolating (linearly).
      current_progress = (i + 1) / len(self._cached_points)
      expected_duration = current_duration / current_progress
      if expected_duration > ten_minutes_threshold:
        logging.warning(
            'Accounting cache is being slow: total duration estimated to'
            ' {%.0fmin} (current progress: %.0f%%)',
            expected_duration / 60,
            100.0 * current_progress,
        )

    end_clock = time.time()
    logging.info(
        'Accounting cache (took %.3fmin).', (end_clock - start_clock) / 60
    )
    self._cache_is_initialized = True

  def compute_epsilon(
      self,
      num_updates: chex.Numeric,
      dp_params: DpParams,
      allow_approximate_cache: bool = False,
  ) -> float:
    """Uses cached results to give an approximate (over-estimated) epsilon.

    The value returned should always be an over-approximation of the true
    epsilon: this method uses the closest `num_updates` in the cache that is
    equal to or greater than the requested `num_updates`. If such a value cannot
    be found, an indexing error will be raised.

    Args:
      num_updates: The number of updates to compute epsilon for.
      dp_params: Parameters required for computing the DP guarnatee.
      allow_approximate_cache: Whether to use the approximate cache.

    Returns:
      Value of epsilon.
    """

    if allow_approximate_cache:
      self._maybe_initialize_cache(dp_params)
      closest_cached_point = self._cached_points[
          _ceil_div(
              self._num_cached_points * num_updates, self._max_num_updates
          )
      ]
      return self._cached_values[closest_cached_point]
    else:
      return self._accountant.compute_epsilon(num_updates, dp_params)
