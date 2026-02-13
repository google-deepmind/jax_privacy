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
