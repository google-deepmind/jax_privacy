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

"""Utilities for calculating deltas in Monte Carlo accounting."""

from typing import Sequence

import numpy as np
import scipy


def _kl(q: float, p: float) -> float:
  return scipy.special.rel_entr(q, p) + scipy.special.rel_entr(1 - q, 1 - p)


def _hoeffding_bound(num_samples: int, tau: float, delta: float) -> float:
  """Upper bound on probability that average of num_samples samples of variable with mean > tau * delta is <= delta, where tau >= 1."""
  # Fact 4.2 of https://arxiv.org/pdf/2412.16802.
  assert tau >= 1
  return np.exp(-num_samples * _kl(delta, tau * delta))


def get_overall_delta(num_samples: int, base_delta: float) -> float:
  """The delta we can formally report using Monte Carlo verification.

  In more detail, if we use num_samples samples to verify that a mechanism
  satisfies (epsilon, base_delta)-DP, then we can formally report that the
  mechanism satisfies (epsilon, overall_delta)-DP.

  This method assumes we are doing many Monte Carlo verifications of different
  ordered mechanisms, and then picking the highest-utility mechanism that meets
  our target privacy guarantee, as in Appendix A of
  https://arxiv.org/abs/2602.09338. If we are only verifying one mechanism
  (which is undesirable in practice), a tighter bound is possible.

  Args:
    num_samples: The number of samples used in Monte Carlo verification.
    base_delta: The base_delta such that we use the samples to verify that each
      mechanism satisfies (epsilon, base_delta)-DP.

  Returns:
    overall_delta such that we can report the end-to-end pipeline of Monte Carlo
    verification and then running the best verified mechanism is (epsilon,
    overall_delta)-DP.
  """
  if base_delta <= 0 or base_delta > 1:
    raise ValueError('base_delta must be in (0, 1].')
  if num_samples <= 0:
    raise ValueError('num_samples must be positive.')

  def overall_delta_from_tau(tau):
    q = _hoeffding_bound(num_samples, tau, base_delta)
    return tau * base_delta + q * (1 - tau * base_delta)

  best_tau = scipy.optimize.minimize_scalar(
      overall_delta_from_tau, bounds=(1, 1 / base_delta), method='bounded'
  ).x
  return min(overall_delta_from_tau(best_tau), 1.0)


def minimum_samples_to_calibrate(base_delta: float, target_delta: float) -> int:
  """The minimum number of samples needed to calibrate to target_delta.

  While for practical values of delta this method is stable, for very small
  target_delta (or base_delta very close to target_delta) this may binary search
  over a number of samples that results in errors due to floats having a maximum
  representable value.

  Args:
    base_delta: The base_delta such that we use Monte Carlo verification to
      check if each mechanism satisfies (epsilon, base_delta)-DP.
    target_delta: The value such that we want to report the overall mechanism of
      Monte Carlo verification and then running the best verified mechanism is
      (epsilon, target_delta)-DP.

  Returns:
    The minimum number of samples needed to calibrate to target_delta.
  """
  if base_delta <= 0 or base_delta > 1:
    raise ValueError('base_delta must be in (0, 1].')
  if target_delta <= 0 or target_delta > 1:
    raise ValueError('target_delta must be in (0, 1].')
  if target_delta <= base_delta:
    raise ValueError('target_delta must be > base_delta.')
  lower_bound = int(1 // target_delta)
  upper_bound = 2 * lower_bound

  # There are some stability issues in going back-and-forth between base_delta
  # and target_delta, so to be conservative we enforce both that we have a valid
  # base_delta and that we achieve the target_delta.
  # TODO: Investigate using log-space to improve stability.
  def _enough_samples(num_samples):
    try:
      get_base_delta(num_samples, target_delta)
      return get_overall_delta(num_samples, base_delta) <= target_delta
    except ValueError:
      return False

  while not _enough_samples(upper_bound):
    lower_bound, upper_bound = 2 * lower_bound, 2 * upper_bound
  while lower_bound < upper_bound - 1:
    mid = (lower_bound + upper_bound) // 2
    if _enough_samples(mid):
      upper_bound = mid
    else:
      lower_bound = mid

  # Due to precision issues, upper_bound might be slightly too small. We fix it
  # post-hoc by increasing it by 1 until it achieves the target_delta and also
  # we have enough samples to find a valid base_delta.
  while not _enough_samples(upper_bound):
    upper_bound += 1
  return upper_bound


def get_base_delta(num_samples: int, target_delta: float) -> float:
  """The base_delta for Monte Carlo verification to achieve target_delta.

  In more detail, if we use num_samples samples to verify that a mechanism
  satisfies (epsilon, base_delta)-DP, then we can formally report that the
  mechanism satisfies (epsilon, target_delta)-DP.

  This method assumes we are doing many Monte Carlo verifications of different
  ordered mechanisms, and then picking the highest-utility mechanism that meets
  our target privacy guarantee. If we are only verifying one mechanism (which
  requires some chance of an empty output which is usually undesirable in
  practice), a tighter bound is possible.

  Args:
    num_samples: The number of samples used in Monte Carlo verification.
    target_delta: The value such that we want to report the overall mechanism of
      Monte Carlo verification and then running the best verified mechanism is
      (epsilon, target_delta)-DP.

  Returns:
    The base_delta such that we use Monte Carlo verification to check if each
    mechanism satisfies (epsilon, base_delta)-DP.
  """
  if num_samples <= 0:
    raise ValueError('num_samples must be positive.')
  if target_delta < 0 or target_delta > 1:
    raise ValueError('target_delta must be in [0, 1].')
  tol = 1e-4 * target_delta
  base_delta = scipy.optimize.minimize_scalar(
      lambda d: abs(get_overall_delta(num_samples, d) - target_delta),
      bounds=(0, target_delta),
      method='bounded',
      options={'xatol': tol},
  ).x
  # Because of the tolerance, we may end up with base_delta that is slightly
  # too small. We report base_delta if it achieves the target_delta, otherwise
  # we try base_delta - tol to be conservative. If that also does not achieve
  # the target_delta, it is possible that the minimum of
  # |overall_delta - target_delta| may be greater than 0, i.e. we cannot find a
  # base_delta that is valid.
  if get_overall_delta(num_samples, base_delta) < target_delta:
    return base_delta
  if get_overall_delta(num_samples, base_delta - tol) < target_delta:
    return base_delta - tol
  raise ValueError(
      'Failed to find a valid base_delta. num_samples may be too small.'
  )


def delta_from_epsilon_and_samples(
    epsilon: float,
    samples: Sequence[float],
    counts: Sequence[float] | None = None,
):
  """Calculate the delta parameter for a given epsilon and list of samples.

  Args:
    epsilon: The epsilon parameter of the DP guarantee.
    samples: Samples from the privacy loss distribution. That is, to estimate
      the e^epsilon-hockey stick divergence between P and Q, we sample x from P,
      and then compute ln(P(x)/Q(x)) to get a single sample.
    counts: A list of floats representing the counts of the samples. If None, we
      assume the samples are count 1. This can be helpful if the number of
      samples is very large, in which case one can discretize the samples and
      pass the counts of each discretization. If passed, should be the same
      length as samples.

  Returns:
    The delta parameter given by Monte Carlo estimation of the hockey-stick
    divergence.
  """
  samples = np.asarray(samples)
  if epsilon < 0:
    raise ValueError('epsilon must be non-negative.')
  if samples.ndim != 1:
    raise ValueError('samples must be a 1D array.')
  if counts is not None:
    counts = np.asarray(counts)
    if counts.ndim != 1:
      raise ValueError('counts must be a 1D array.')
    if samples.size != counts.size:
      raise ValueError('samples and counts must have the same size.')
  np_min = np.minimum
  return np.average(-np.expm1(np_min(epsilon - samples, 0.0)), weights=counts)


def perform_calibration_from_samples(
    epsilon: float,
    delta: float,
    *,
    positive_samples: Sequence[Sequence[float]],
    positive_counts: Sequence[Sequence[float]] | None = None,
    negative_samples: Sequence[Sequence[float]] | None = None,
    negative_counts: Sequence[Sequence[float]] | None = None,
) -> tuple[bool, int | float]:
  """Perform calibration to find the highest-utility hyperparameter satisfying the target DP guarantee.

  This is Algorithm 5 in https://arxiv.org/pdf/2602.09338.

  In particular, this method assumes we have chosen a sweep of a hyperparameter
  (such as noise multiplier), for each value in the sweep independently
  generated some number of samples, and now want to find the highest-utility
  hyperparameter (e.g. the lowest noise multiplier or highest number of training
  iterations in DP-SGD) we can use and still report end-to-end
  (epsilon, delta)-DP using Monte Carlo verification.

  We assume that the samples are ordered from highest to lowest privacy of the
  hyperparameters, i.e. lowest to highest delta at a fixed epsilon
  (or vice-versa), even though the privacy parameters may not be explicitly
  known. e.g., in the example of calibrating the noise multiplier for
  DP-BandMF, we would order the samples from highest to lowest noise multiplier,
  even if we may not know the exact (epsilon, delta)-DP guarantee for each noise
  multiplier.

  If different numbers of samples were used for each hyperparameter, this method
  conservatively uses the minimum number of samples across all values of the
  hyperparameter for computing tail bounds.

  Example Usage (calibrating a Gaussian mechanism):
    >>> epsilon, delta = 4.0, 1e-3
    >>> nm_sweep = [2.0, 1.0, 0.5]
    >>> positive_samples = []
    >>> for nm in nm_sweep:
    ...   positive_samples.append(
    ...     np.random.normal(loc=0.5 / nm ** 2, size=1_000_000)
    ...   )
    >>> passes_verification, best_nm_index = perform_calibration_from_samples(
    ...   epsilon, delta, positive_samples=positive_samples
    ... )
    >>> if passes_verification:
    ...   print(nm_sweep[best_nm_index])
    1.0

  Args:
    epsilon: The epsilon parameter of the target DP guarantee.
    delta: The delta parameter of the target DP guarantee.
    positive_samples: A list of lists of privacy loss samples from the positive
      case, one list for each hyperparameter value.
    positive_counts: An optional list of lists of counts corresponding to each
      of the lists of positive samples. If None, we assume the samples are count
      1. If passed, each list should be the same length as the associated list
      of positive_samples.
    negative_samples: An optional list of lists of privacy loss samples from the
      negative case, one list for each hyperparameter value. If None, this means
      we assume the positive case has a worse DP guarantee than the negative
      case, so we only need to do accounting for the positive case.
    negative_counts: An optional list of lists of counts corresponding to each
      of the lists of negative samples. If None, we assume the samples are
      unweighted. If passed, each list should be the same length as the
      associated list of negative_samples. Ignored if negative_samples is None.

  Returns:
    Either (True, i), or (False, base_delta). If there is an associated set of
    hyperparameters and (True, i) is returned, i is the index of the
    highest-utility hyperparameter we can use while satisfying the target
    (epsilon, delta)-DP guarantee. If (False, base_delta) is returned, we should
    fall back to a mechanism that is known to be (epsilon, base_delta)-DP (i.e.,
    one that can be calibrated without Monte Carlo verification)
  """
  if not positive_samples:
    raise ValueError('positive_samples must be non-empty.')
  if positive_counts is not None and len(positive_samples) != len(
      positive_counts
  ):
    raise ValueError(
        'positive_samples and positive_counts must have the same length.'
    )
  if negative_samples is not None and len(positive_samples) != len(
      negative_samples
  ):
    raise ValueError(
        'positive_samples and negative_samples must have the same length.'
    )
  if negative_counts is not None and len(positive_samples) != len(
      negative_counts
  ):
    raise ValueError(
        'positive_samples and negative_counts must have the same length.'
    )
  if positive_counts is None:
    positive_counts = [np.ones_like(samples) for samples in positive_samples]
  if negative_samples is not None and negative_counts is None:
    negative_counts = [np.ones_like(samples) for samples in negative_samples]
  positive_sample_counts = [sum(counts) for counts in positive_counts]
  if negative_samples is None:
    negative_sample_counts = []
  else:
    negative_sample_counts = [sum(counts) for counts in negative_counts]
  min_sample_count = min(positive_sample_counts + negative_sample_counts)

  base_delta = get_base_delta(min_sample_count, delta)

  first_positive_delta = delta_from_epsilon_and_samples(
      epsilon, positive_samples[0], positive_counts[0]
  )
  if negative_samples is None:
    first_negative_delta = 0.0
  else:
    first_negative_delta = delta_from_epsilon_and_samples(
        epsilon, negative_samples[0], negative_counts[0]
    )
  if first_positive_delta > base_delta or first_negative_delta > base_delta:
    return False, base_delta

  for i in range(1, len(positive_samples)):
    positive_delta = delta_from_epsilon_and_samples(
        epsilon, positive_samples[i], positive_counts[i]
    )
    if negative_samples is None:
      negative_delta = 0.0
    else:
      negative_delta = delta_from_epsilon_and_samples(
          epsilon, negative_samples[i], negative_counts[i]
      )
    if positive_delta > base_delta or negative_delta > base_delta:
      # This hyperparameter does not pass verification, return the previous one.
      return True, i - 1
  return True, len(positive_samples) - 1
