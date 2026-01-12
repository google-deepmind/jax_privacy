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

"""Library for empirical privacy auditing/estimation.

This library provides functions for estimating the privacy of a model,
based on attack scores of held-in and held-out canaries.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import functools
from typing import Callable, TypeAlias

import dp_accounting
import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats


_norm = scipy.stats.norm
_binom = scipy.stats.binom
_logistic = scipy.special.expit
_brentq = scipy.optimize.brentq


class ThresholdStrategy:
  """Base class for threshold selection strategies."""


class Bonferroni(ThresholdStrategy):
  """Use Bonferroni correction across all possible thresholds."""


@dataclasses.dataclass(frozen=True)
class Explicit(ThresholdStrategy):
  """Use a specific threshold value.

  Attributes:
    threshold: The threshold value to use.
  """

  threshold: float


@dataclasses.dataclass(frozen=True)
class Split(ThresholdStrategy):
  """Split data to choose threshold and then compute the bound.

  Attributes:
    threshold_estimation_frac: The fraction of data to use for computing the
      threshold. The rest will be used for computing the bound.
    seed: The seed for the random number generator. If None, a seed will be
      chosen non-deterministically.
  """

  threshold_estimation_frac: float = 0.5
  seed: int | None = None


AuditAllThresholdsMethod: TypeAlias = Callable[
    [
        'CanaryScoreAuditor',  # self
        float,  # significance
        float,  # delta
        bool,  # one_sided
        float | None,  # threshold or None for all thresholds
    ],
    tuple[float, float] | float,
]


def _one_run_p_value(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
  """Computes p-value for one-shot audit.

  See https://arxiv.org/pdf/2305.08846 for details.

  Args:
    m: The number of canaries.
    n_guess: The number of guesses.
    n_correct: The number of correct guesses.
    eps: The epsilon value to compute the p-value for.
    delta: The delta value to compute the p-value for.

  Returns:
    The p-value for the one-shot audit.
  """
  q = _logistic(eps)
  beta = _binom.sf(n_correct - 1, n_guess, q)
  if delta == 0:
    return beta

  i_vals = np.arange(1, n_correct + 1)
  cum_sums = _binom.sf(n_correct - i_vals - 1, n_guess, q) - beta
  alpha = np.max(cum_sums / i_vals, initial=0)

  return min(beta + alpha * delta * 2 * m, 1)


def _epsilon_one_run(
    eps_lo: float,
    m: int,
    n_guess: int,
    n_correct: int,
    significance: float,
    delta: float,
) -> float:
  """Computes epsilon bound for one-shot audit for a given n_guess/n_correct.

  See https://arxiv.org/pdf/2305.08846 for details.

  Args:
    eps_lo: Smallest epsilon value to consider.
    m: The number of canaries.
    n_guess: The number of guesses.
    n_correct: The number of correct guesses.
    significance: The significance level.
    delta: DP delta parameter.

  Returns:
    The highest epsilon for which the (epsilon, delta) claim is falsified, or
    eps_lo if the claim is not falsified for any eps >= eps_lo.
  """
  if _one_run_p_value(m, n_guess, n_correct, eps_lo, 0) > significance:
    # Cheap fail fast: If even with delta=0 we can't reject the null at eps_lo,
    # we certainly can't reject it with delta > 0.
    return eps_lo

  def audit_objective(eps):
    # Returns a positive value if the (epsilon, delta) claim is falsified.
    return significance - _one_run_p_value(m, n_guess, n_correct, eps, delta)

  if audit_objective(eps_lo) <= 0:
    return eps_lo

  eps_hi = max(1.0, 2 * eps_lo)
  while audit_objective(eps_hi) > 0:
    eps_lo, eps_hi = eps_hi, eps_hi * 2

  return _brentq(audit_objective, eps_lo, eps_hi, xtol=1e-6)


@functools.lru_cache(maxsize=1000)
def _gaussian_dp_blow_up_inverse(
    eps: float, delta: float
) -> Callable[[float], float]:
  """Computes the inverse of the Gaussian differential privacy blow-up."""
  sigma = dp_accounting.get_sigma_gaussian(eps, delta)
  if sigma == 0:
    return lambda x: float(x == 1.0)
  else:
    return lambda x: _norm.cdf(_norm.ppf(x) - 1 / sigma)


def _epsilon_one_run_fdp(
    eps_lo: float,
    m: int,
    n_guess: int,
    n_correct: int,
    significance: float,
    delta: float,
) -> float:
  """Computes the epsilon bound for a given number of true and false positives.

  See https://arxiv.org/pdf/2410.22235 for details.

  Args:
    eps_lo: Smallest epsilon value to consider.
    m: The number of canaries.
    n_guess: The number of guesses.
    n_correct: The number of correct guesses.
    significance: The significance level.
    delta: DP delta parameter.

  Returns:
    The highest epsilon for which the (epsilon, delta) claim is falsified, or
    eps_lo if the claim is not falsified for any eps >= eps_lo.
  """

  def audit_objective(eps):
    # Returns a positive value if attack scores falsify the f-DP claim.
    # See 'audit_rh_with_cap' in https://arxiv.org/pdf/2410.22235.
    blow_up_inv_fn = _gaussian_dp_blow_up_inverse(eps, delta)

    r = significance * n_correct / m
    h = significance * (n_guess - n_correct) / m

    for i in range(n_correct - 1, -1, -1):
      h_new = max(h, blow_up_inv_fn(r))
      if h == h_new:
        # If h is unchanged, then neither h nor r can ever change afterward.
        break
      r_new = min(r + (i / (n_guess - i)) * (h_new - h), 1.0)
      h, r = h_new, r_new

    return r + h - n_guess / m

  if audit_objective(eps_lo) <= 0:
    return eps_lo

  eps_hi = max(1.0, 2 * eps_lo)
  while audit_objective(eps_hi) > 0:
    eps_lo, eps_hi = eps_hi, eps_hi * 2

  return _brentq(audit_objective, eps_lo, eps_hi, xtol=1e-6)


@dataclasses.dataclass(frozen=True)
class BootstrapParams:
  """Parameters for bootstrapping.

  Several methods in this library that compute a privacy metric (e.g., AUROC,
  epsilon) optionally perform bootstrapping to estimate quantiles of the metric.
  The held-in and held-out canary scores are resampled with replacement, and
  metric is estimated from the resampled data. Repeating this sampling procedure
  many times produces an empirical distribution of the metric, from which we can
  estimate quantiles. This class configures the number of resamples, the
  quantiles to report, and the seed for the random number generator.

  Attributes:
    num_samples: The number of times to resample the canary scores.
    quantiles: An array-like of quantiles to report. E.g., for a 95% confidence
      interval, use (0.025, 0.975).
    seed: The seed for the random number generator.
  """

  num_samples: int = 1000
  quantiles: np.typing.ArrayLike = (0.025, 0.975)
  seed: int | None = None

  def __post_init__(self):
    quantile_arr = np.asarray(self.quantiles)
    if quantile_arr.size == 0:
      raise ValueError('quantiles cannot be empty.')
    if not np.all((0 < quantile_arr) & (quantile_arr < 1)):
      raise ValueError(f'quantiles must be in (0, 1), got {self.quantiles}.')

  @classmethod
  def confidence_interval(
      cls,
      num_samples: int = 1000,
      confidence: float = 0.95,
      seed: int | None = None,
  ) -> BootstrapParams:
    """Creates a BootstrapParams object for a confidence interval.

    Args:
      num_samples: The number of times to resample the canary scores.
      confidence: The desired confidence level.
      seed: The seed for the random number generator.

    Returns:
      A BootstrapParams object for computing a confidence interval.
    """
    if not 0 < confidence < 1:
      raise ValueError(f'confidence must be in (0, 1), got {confidence}.')
    significance = 1 - confidence
    quantiles = (significance / 2, 1 - significance / 2)
    return cls(num_samples=num_samples, quantiles=quantiles, seed=seed)


def _log_sub(x, y):
  # Stable computation of log(exp(x) - exp(y)).
  if np.any(y > x):
    raise ValueError(f'y must be less than or equal to x, got y={y} and x={x}.')
  with np.errstate(divide='ignore'):  # Okay to return -np.inf if x == y.
    return x + np.log1p(-np.exp(y - x))


def _clopper_pearson_upper(
    k: int | np.ndarray, n: int, significance: float
) -> np.ndarray:
  """Computes Clopper-Pearson one-sided upper binomial confidence interval.

  Args:
    k: The number of successes.
    n: The number of trials.
    significance: Allowed probability of failure (one minus confidence).

  Returns:
    A value p such that the probability of observing k or fewer successes out of
    n Bernoulli(p) trials is approximately significance.
  """
  return np.where(
      k < n, scipy.stats.beta.ppf(1 - significance, k + 1, n - k), 1.0
  )


def _pareto_frontier(points: np.ndarray) -> np.ndarray:
  """Computes the indices of the pareto frontier of a piecewise linear function.

  Given a piecewise linear function defined by a sequence of points, computes
  the set of points that are not weakly linearly dominated by any pair of outer
  points. That is, given a list of points (sorted by x coordinate), retain only
  points (x_i, y_i) for which there do not exist j < i and k > i and number a
  with 0 <= a <= 1 such that x_i = (1-a)x_j + ax_k and y_i <= (1-a)y_j + ay_k.

  The algorithm iteratively discards points that are dominated by their
  neighbors. Each iteration uses fast vectorized operations. It could be O(N^2)
  in pathological cases, but in practice it is very fast. If it needs to be
  optimized, Gemini suggests using a block-based divide and conquer approach:
  run the algorithm on blocks of about 1000 points and then run it again on the
  merged results.

  Args:
    points: An array of shape (N, 2) with N >= 2 containing the vertices
      defining the segments of the piecewise linear function, sorted by x
      coordinate.

  Returns:
    A numpy array of length M with 2 <= M <= N, containing the indices of the
    vertices on the pareto frontier.
  """
  if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
    raise ValueError(f'Expected at least two 2D points, got {points.shape}.')

  if not np.all(points[:-1, 0] <= points[1:, 0]):
    raise ValueError('Expected points to be sorted by x-coordinate.')

  indices = np.arange(points.shape[0])
  while True:
    if len(indices) <= 2:
      break

    diff = np.diff(points[indices], axis=0)
    cross_product = diff[:-1, 1] * diff[1:, 0] - diff[1:, 1] * diff[:-1, 0]
    dominated_mask = cross_product <= 0

    # If no points are dominated in this pass, the hull is stable.
    if not np.any(dominated_mask):
      break

    keep_mask = np.r_[True, ~dominated_mask, True]
    indices = indices[keep_mask]

  return indices


def _get_tn_fn_counts(
    in_canary_scores: Sequence[float],
    out_canary_scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes true negative and false negative counts at each threshold.

  Args:
    in_canary_scores: Attack scores of held-in canaries.
    out_canary_scores: Attack scores of held-out canaries.

  Returns:
    A tuple (thresholds, true negative counts, false negative counts) for each
    threshold.
  """
  in_scores = np.asarray(in_canary_scores)
  out_scores = np.asarray(out_canary_scores)

  if in_scores.size == 0 and out_scores.size == 0:
    raise ValueError(
        'At least one of the canary score arrays should be non-empty.'
    )

  # Get the unique, sorted thresholds from both arrays.
  unique_scores_sorted = np.union1d(in_scores, out_scores)

  # Append np.inf to ensure thresholds > max_score are considered.
  thresholds = np.concatenate((unique_scores_sorted, [np.inf]))

  # Sort the score arrays for faster search.
  in_sorted = np.sort(in_scores)
  out_sorted = np.sort(out_scores)

  # For each threshold, find the index where it would be inserted to maintain
  # order. With side='left', this index is the first location with a value >=
  # threshold.
  fn_counts = np.searchsorted(in_sorted, thresholds, side='left')
  tn_counts = np.searchsorted(out_sorted, thresholds, side='left')

  counts = np.stack([fn_counts, tn_counts], axis=1)
  indices = _pareto_frontier(counts)
  return thresholds[indices], tn_counts[indices], fn_counts[indices]


def _tpr_at_given_fpr(
    fpr: np.typing.ArrayLike,
    tn_counts: np.ndarray,
    fn_counts: np.ndarray,
) -> np.ndarray | float:
  """Computes maximum TPR at a given FPR."""
  fpr = np.asarray(fpr)

  if not np.all((0 <= fpr) & (fpr <= 1)):
    raise ValueError(f'fpr must be in [0, 1], got {fpr}.')

  # This implementation assumes that tn_counts and fn_counts are sorted in
  # increasing order, and have been filtered to contain only thresholds on the
  # boundary of the convex hull.
  n_pos = fn_counts[-1]
  n_neg = tn_counts[-1]

  target_tn_count = n_neg * (1 - fpr)

  # Find the first threshold greater or equal to target_tn_count. We will
  # interpolate between the previous one and this one.
  threshold = np.searchsorted(tn_counts, target_tn_count)

  # If threshold is 0 (i.e., if target_tn_count is 0), we interpolate between
  # the first two thresholds. Because we have found the convex hull of FN/TN
  # counts, the first TN count is always 0 and the second is positive.
  threshold = np.maximum(threshold, 1)

  tn_left = tn_counts[threshold - 1]
  tn_right = tn_counts[threshold]
  p = (target_tn_count - tn_left) / (tn_right - tn_left)

  fn_left = fn_counts[threshold - 1]
  fn_right = fn_counts[threshold]
  fnr = ((1 - p) * fn_left + p * fn_right) / n_pos

  return 1 - fnr


def _epsilon_raw_counts_helper(
    tn_counts: np.ndarray,
    fn_counts: np.ndarray,
    min_count: int,
    delta: float,
) -> float:
  """Estimates epsilon given true and false counts at each threshold."""
  n_pos = fn_counts[-1]
  n_neg = tn_counts[-1]

  if min_count >= n_neg:
    return 0.0

  min_fpr = min_count / n_neg
  tpr_at_min_fpr = _tpr_at_given_fpr(min_fpr, tn_counts, fn_counts)

  if delta == 0:
    return np.log(tpr_at_min_fpr / min_fpr)

  tp_counts = n_pos - fn_counts
  fp_counts = n_neg - tn_counts

  pos_counts = np.stack([tp_counts, fp_counts], axis=1)
  pos_counts = pos_counts[fp_counts >= min_count]
  pos_rates = pos_counts / [n_pos, n_neg]
  pos_rates = np.append(pos_rates, [[tpr_at_min_fpr, min_fpr]], axis=0)

  # We want to ignore invalid values in the log here. If (tpr - delta) is
  # less or equal to zero, the bound is invalid. Let it return np.nan or -np.inf
  # and we will filter it with np.nanmax.
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.nanmax(
        np.log(pos_rates[:, 0] - delta) - np.log(pos_rates[:, 1]),
        initial=0,
    )


def _random_partition(
    scores: np.ndarray,
    rng: np.random.Generator,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Randomly splits a score array into two parts."""
  if not 0 < p < 1:
    raise ValueError(f'p must be in (0, 1), got {p}.')

  perm = rng.permutation(len(scores))
  split_idx = int(len(scores) * p)
  return scores[perm[:split_idx]], scores[perm[split_idx:]]


class CanaryScoreAuditor:
  """Class for auditing privacy based on attack scores.

  To use this library, create a CanaryScoreAuditor providing the attack scores
  of held-in and held-out canaries. Attack scores can be any value such that
  that held-in canaries are expected to have higher scores, for example the
  log-likelihood or the likelihood ratio to the pretrained model. Then the
  auditor can be used to compute privacy metrics, including various epsilon
  lower bounds from the literature, the maximum TPR at a given FPR, and the area
  under the receiver operating characteristic (ROC) curve.

  Example Usage:
    >>> out_canary_scores = np.arange(100)
    >>> in_canary_scores = np.arange(100) + 9.5
    >>> auditor = CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    >>> tpr_at_low_fpr = auditor.tpr_at_given_fpr(0.01)
    >>> float(round(tpr_at_low_fpr, 2))
    0.11
    >>> auroc = auditor.attack_auroc()
    >>> float(auroc)
    0.595
  """

  def __init__(
      self,
      in_canary_scores: Sequence[float],
      out_canary_scores: Sequence[float],
  ):
    """Initializes the CanaryScoreAuditor.

    IMPORTANT: We consider decision rules that classify examples that score
    *higher* than the threshold as "in". If held-in canaries are expected to
    have lower scores than held-out canaries, then negate the score before
    constructing the auditor.

    Args:
      in_canary_scores: Attack scores of held-in canaries.
      out_canary_scores: Attack scores of held-out canaries.
    """
    self._in_canary_scores = np.array(in_canary_scores)
    self._out_canary_scores = np.array(out_canary_scores)
    if self._in_canary_scores.size == 0:
      raise ValueError('in_canary_scores must be non-empty.')
    if self._out_canary_scores.size == 0:
      raise ValueError('out_canary_scores must be non-empty.')

    self._thresholds, self._tn_counts, self._fn_counts = _get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )

  def _bootstrap(
      self,
      fn: Callable[['CanaryScoreAuditor'], float],
      params: BootstrapParams,
  ) -> np.ndarray:
    """Computes bootstrapped quantiles for a function.

    Args:
      fn: A function of a CanaryScoreAuditor returning a scalar.
      params: The parameters for the bootstrap.

    Returns:
      An array of bootstrapped quantiles.
    """
    rng = np.random.default_rng(seed=params.seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=params.num_samples)

    def get_value(seed):
      inner_rng = np.random.default_rng(seed=seed)
      in_samples = inner_rng.choice(
          self._in_canary_scores, size=self._in_canary_scores.size
      )
      out_samples = inner_rng.choice(
          self._out_canary_scores, size=self._out_canary_scores.size
      )
      return fn(CanaryScoreAuditor(in_samples, out_samples))

    with futures.ThreadPoolExecutor() as pool:
      values = list(pool.map(get_value, seeds))

    return np.quantile(values, params.quantiles)

  def _epsilon_clopper_pearson_all_thresholds(
      self,
      significance: float,
      delta: float,
      one_sided: bool,
      threshold: float | None = None,
  ) -> tuple[float, float] | float:
    """Estimates epsilon with C-P bound at one or all thresholds."""
    if threshold is None:
      fn_counts = self._fn_counts
      tn_counts = self._tn_counts
    else:
      fn_counts = np.array([np.sum(self._in_canary_scores < threshold)])
      tn_counts = np.array([np.sum(self._out_canary_scores < threshold)])

    n_pos = self._fn_counts[-1]
    n_neg = self._tn_counts[-1]

    fnr_ubs = _clopper_pearson_upper(fn_counts, n_pos, significance / 2)
    fp_counts = n_neg - tn_counts
    fpr_ubs = _clopper_pearson_upper(fp_counts, n_neg, significance / 2)

    def eps_and_idx(fnr_ubs, fpr_ubs):
      tpr_lbs = 1 - fnr_ubs
      valid = np.flatnonzero(tpr_lbs > delta)
      if valid.size == 0:
        return 0.0, -1
      eps_vals = np.log(tpr_lbs[valid] - delta) - np.log(fpr_ubs[valid])
      subidx = np.argmax(eps_vals)
      return eps_vals[subidx], valid[subidx]

    eps, idx = eps_and_idx(fnr_ubs, fpr_ubs)

    if not one_sided:
      new_eps, new_idx = eps_and_idx(fpr_ubs, fnr_ubs)
      if new_eps > eps:
        idx = new_idx
        eps = new_eps

    if threshold is None:
      return eps, self._thresholds[idx]
    else:
      return eps

  def epsilon_clopper_pearson(
      self,
      significance: float,
      delta: float = 0,
      one_sided: bool = True,
      *,
      threshold_strategy: ThresholdStrategy = Bonferroni(),
  ) -> float:
    """Finds epsilon lower bound from scores of held-in/held-out canaries.

    Described in https://arxiv.org/pdf/2101.04535.

    Args:
      significance: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta.
      one_sided: Whether to use only TPR/FPR (vs. max of TPR/FPR and TNR/FNR).
      threshold_strategy: How to select the threshold to use for the epsilon
        estimate.

    Returns:
      Optimal epsilon lower bound.
    """
    if not 0 < significance < 0.5:
      raise ValueError(f'significance must be in (0, 0.5), got {significance}.')
    if not 0 <= delta <= 1:
      raise ValueError(f'delta must be in [0, 1], got {delta}.')

    return self._audit_with_threshold_strategy(
        threshold_strategy,
        CanaryScoreAuditor._epsilon_clopper_pearson_all_thresholds,
        significance,
        delta,
        one_sided,
    )

  def epsilon_raw_counts(
      self,
      min_count: int = 50,
      delta: float = 0,
      one_sided: bool = True,
      *,
      bootstrap_params: BootstrapParams | None = None,
  ) -> float | np.ndarray:
    """Estimates epsilon from raw count statistics of seen/unseen canaries.

    `min_count` is the minimum number of FP (or FN, if not one_sided) required
    to consider a threshold. If `min_count` is too high relative to the number
    of canaries, the estimate will be biased towards zero. If it is too low, the
    estimate will have high variance.

    Args:
      min_count: Only consider thresholds with this many TP/FP (TN/FN).
      delta: Approximate DP delta.
      one_sided: Whether to use only TPR/FPR (vs. max of TPR/FPR and TNR/FNR).
      bootstrap_params: If provided, compute and return bootstrapped quantiles
        of the estimate. Note that this should not be interpreted as a formal
        confidence interval on the true epsilon, merely a confidence interval of
        the estimate.

    Returns:
      Epsilon estimate ln(TPR/FPR).
    """
    if min_count < 0:
      raise ValueError(f'min_count must be non-negative, got {min_count}.')
    if not 0 <= delta <= 1:
      raise ValueError(f'delta must be in [0, 1], got {delta}.')

    if bootstrap_params is not None:
      return self._bootstrap(
          lambda a: a.epsilon_raw_counts(min_count, delta, one_sided),
          bootstrap_params,
      )

    eps = _epsilon_raw_counts_helper(
        self._tn_counts, self._fn_counts, min_count, delta
    )

    if not one_sided:
      # For a two-sided estimate, we also consider the reverse decision rule,
      # where examples with scores *lower* than the threshold are classified as
      # "in". Reverse the counts because the helper function expects the counts
      # to be in ascending order.

      tp_counts = (self._fn_counts[-1] - self._fn_counts)[::-1]
      fp_counts = (self._tn_counts[-1] - self._tn_counts)[::-1]
      eps = max(
          eps,
          _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count, delta),
      )

    return max(eps, 0)

  def tpr_at_given_fpr(
      self,
      fpr: np.typing.ArrayLike,
      *,
      bootstrap_params: BootstrapParams | None = None,
  ) -> np.ndarray | float:
    """Computes maximum TPR at a given FPR.

    Args:
      fpr: The desired false positive rate. May be a scalar, or an array of
        independent FPR values, in which case an array of the same shape is
        returned with the TPR at each FPR.
      bootstrap_params: If provided, compute and return bootstrapped quantiles
        of the TPR. `fpr` must be a scalar in this case.

    Returns:
      The maximum true positive rate at the given false positive rate,
      allowing classifiers that randomize between two thresholds.
    """
    fpr = np.asarray(fpr)
    if not np.all((0 <= fpr) & (fpr <= 1)):
      raise ValueError(f'fpr must be in [0, 1], got {fpr}.')

    if bootstrap_params is not None:
      if fpr.ndim > 0:
        raise ValueError(
            f'fpr must be a scalar for bootstrap, got shape {fpr.shape}.'
        )
      return self._bootstrap(
          lambda auditor: auditor.tpr_at_given_fpr(fpr),
          bootstrap_params,
      )

    return _tpr_at_given_fpr(fpr, self._tn_counts, self._fn_counts)

  def attack_auroc(
      self,
      *,
      bootstrap_params: BootstrapParams | None = None,
  ) -> float | np.ndarray:
    """Computes the area under the ROC curve from the attack scores.

    Args:
      bootstrap_params: If provided, compute and return bootstrapped quantiles
        of the AUROC.

    Returns:
      The area under the ROC curve from the attack scores, allowing classifiers
      that randomize between two thresholds.
    """
    if bootstrap_params is not None:
      return self._bootstrap(CanaryScoreAuditor.attack_auroc, bootstrap_params)

    # Calculate AUROC using the trapezoidal rule. Since we have TN counts and FN
    # counts handy, we're actually computing the area under the curve of the TNR
    # as a function of FNR, which is equivalent.
    tnr = self._tn_counts / self._tn_counts[-1]
    fnr = self._fn_counts / self._fn_counts[-1]
    return 0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1])

  def max_accuracy(
      self,
      *,
      prevalence: float | None = None,
      significance: float | None = None,
  ) -> float:
    """Computes the maximum accuracy achievable by a threshold-based classifier.

    Args:
      prevalence: The prevalence of the positive class. If not provided, the
        prevalence is taken to be the proportion of in-canary examples to the
        total.
      significance: If provided, compute and return the high probability upper
        bound on the maximum accuracy with this allowable probability of failure
        (one minus confidence).

    Returns:
      The maximum accuracy.
    """
    n_pos = self._fn_counts[-1]
    n_neg = self._tn_counts[-1]

    if prevalence is None:
      prevalence = n_pos / (n_pos + n_neg)

    tp_counts = n_pos - self._fn_counts
    if significance is None:
      tnr_ubs = self._tn_counts / n_neg
      tpr_ubs = tp_counts / n_pos
    else:
      tnr_ubs = _clopper_pearson_upper(self._tn_counts, n_neg, significance / 2)
      tpr_ubs = _clopper_pearson_upper(tp_counts, n_pos, significance / 2)

    return np.max(tpr_ubs * prevalence + tnr_ubs * (1 - prevalence))

  def epsilon_from_gdp(
      self,
      significance: float,
      delta: float,
      eps_tol: float = 1e-6,
  ) -> float:
    """Calculates the an estimate for epsilon with GDP.

    This is the method used in https://arxiv.org/pdf/2302.07956 and described in
    https://arxiv.org/pdf/2406.04827.

    Args:
      significance: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta. Must be in (0, 1].
      eps_tol: The tolerance for epsilon (the privacy parameter). Defaults to
        1e-6.

    Returns:
        The estimated epsilon.
    """
    if not 0 < significance < 0.5:
      raise ValueError(f'significance must be in (0, 0.5), got {significance}.')
    if not 0 < delta <= 1:
      raise ValueError(f'delta must be in (0, 1], got {delta}.')
    if eps_tol <= 0:
      raise ValueError(f'eps_tol must be positive, got {eps_tol}.')

    n_pos = self._fn_counts[-1]
    n_neg = self._tn_counts[-1]

    n = len(self._fn_counts)

    # Apply Bonferroni correction over 2 * n hypotheses.
    fnr_ubs = _clopper_pearson_upper(
        self._fn_counts, n_pos, significance / (2 * n)
    )
    fp_counts = n_neg - self._tn_counts
    fpr_ubs = _clopper_pearson_upper(fp_counts, n_neg, significance / (2 * n))

    bounds = np.stack([fpr_ubs, fnr_ubs], axis=1)

    # Filter any thresholds where TNR or TPR is too small.
    bounds = bounds[np.max(bounds, axis=1) < 1 - delta]
    if not bounds.size:
      return 0

    # Eq. 6 in https://arxiv.org/abs/1905.02383. If FPR + FNR is too large,
    # the bound still holds in reverse (by switching D and D'), which has the
    # effect of making mu from Eq. 6 negative. Hence we look for the maximum
    # absolute value of mu.
    max_mu = np.max(np.abs(_norm.isf(bounds[:, 0]) - _norm.ppf(bounds[:, 1])))
    if max_mu == 0:
      return 0

    # Conversion of GDP to (eps, delta)-DP. Corollary 2.13 in
    # https://arxiv.org/abs/1905.02383, in the log domain for stability.
    def delta_gap(eps):
      return _log_sub(
          _norm.logcdf(-(eps / max_mu) + max_mu / 2),
          eps + _norm.logcdf(-(eps / max_mu) - max_mu / 2),
      ) - np.log(delta)

    eps_lb, eps_ub = 0, 100
    if delta_gap(eps_lb) <= 0:
      return eps_lb
    if delta_gap(eps_ub) >= 0:
      return eps_ub

    return _brentq(delta_gap, eps_lb, eps_ub, xtol=eps_tol)

  def _epsilon_one_run_all_thresholds(
      self,
      significance: float,
      delta: float,
      one_sided: bool,
      threshold: float | None = None,
      use_fdp: bool = False,
  ) -> float | tuple[float, float]:
    """Computes the epsilon bound at one or all thresholds."""
    if not one_sided:
      raise ValueError('one_sided must be True.')

    if use_fdp:
      audit_fn = _epsilon_one_run_fdp
    else:
      audit_fn = _epsilon_one_run

    n_pos = self._fn_counts[-1]
    n_neg = self._tn_counts[-1]
    m = n_pos + n_neg

    if threshold is not None:
      tp = np.sum(self._in_canary_scores >= threshold)
      fp = np.sum(self._out_canary_scores >= threshold)
      return audit_fn(0, m, tp + fp, tp, significance, delta)

    tp_counts = n_pos - self._fn_counts
    fp_counts = n_neg - self._tn_counts

    # As a heuristic, search in order of decreasing precision bound, so later
    # thresholds are likely to be ruled out without a full search over eps.
    total_counts = tp_counts + fp_counts
    prec_lbs = 1.0 - _clopper_pearson_upper(
        fp_counts, total_counts, significance
    )
    sorted_indices = np.argsort(-prec_lbs)

    best_eps = 0
    best_idx = -1
    best_q = 0.5

    for idx in sorted_indices:
      n_guess = total_counts[idx]
      n_correct = tp_counts[idx]

      if n_guess == 0 or n_correct / n_guess <= best_q:
        # Raw precision at this threshold is already less than the best so far.
        continue

      new_eps = audit_fn(best_eps, m, n_guess, n_correct, significance, delta)
      if new_eps > best_eps:
        best_eps = new_eps
        best_idx = idx
        best_q = _logistic(best_eps)

    return best_eps, self._thresholds[best_idx]

  def epsilon_one_run(
      self,
      significance: float,
      delta: float,
      one_sided: bool = True,
      *,
      threshold_strategy: ThresholdStrategy = Bonferroni(),
  ) -> float:
    r"""Computes lower bound on epsilon for a single round of auditing.

    This is an implementation of the method from Steinke et al. 2024, "Privacy
    Auditing in One (1) Training Run": https://arxiv.org/abs/2305.08846.

    Currently only one-sided hypotheses are supported ($k_- = 0$).

    Args:
      significance: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta.
      one_sided: Whether to consider only hypotheses with ($k_- = 0$). Must be
        True.
      threshold_strategy: How to select the threshold to use for the epsilon
        estimate.

    Returns:
      The estimated epsilon lower bound.
    """
    if not 0 < significance < 1.0:
      raise ValueError(f'significance must be in (0, 1.0), got {significance}.')
    if not 0 < delta <= 1:
      raise ValueError(f'delta must be in (0, 1], got {delta}.')
    if not one_sided:
      raise ValueError('one_sided must be True.')

    return self._audit_with_threshold_strategy(
        threshold_strategy,
        CanaryScoreAuditor._epsilon_one_run_all_thresholds,
        significance,
        delta,
        one_sided,
    )

  def epsilon_one_run_fdp(
      self,
      significance: float,
      delta: float,
      one_sided: bool = True,
      *,
      threshold_strategy: ThresholdStrategy = Bonferroni(),
  ) -> float:
    """Computes lower bound on epsilon for a single round of auditing.

    This is an implementation of the method from Mahloujifar et al. 2024,
    "Auditing f-Differential Privacy in One Run":
    https://arxiv.org/pdf/2410.22235.

    Currently only one-sided hypotheses are supported ($k_- = 0$).

    Args:
      significance: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta.
      one_sided: Whether to consider only hypotheses with ($k_- = 0$). Must be
        True.
      threshold_strategy: How to select the threshold to use for the epsilon
        estimate.

    Returns:
      The estimated epsilon lower bound.
    """
    if not 0 < significance < 1.0:
      raise ValueError(f'significance must be in (0, 1.0), got {significance}.')
    if not 0 < delta <= 1:
      raise ValueError(f'delta must be in (0, 1], got {delta}.')
    if not one_sided:
      raise ValueError('one_sided must be True.')

    return self._audit_with_threshold_strategy(
        threshold_strategy,
        functools.partial(
            CanaryScoreAuditor._epsilon_one_run_all_thresholds,
            use_fdp=True,
        ),
        significance,
        delta,
        one_sided,
    )

  def _audit_with_threshold_strategy(
      self,
      threshold_strategy: ThresholdStrategy,
      audit_all_thresholds_method: AuditAllThresholdsMethod,
      significance: float,
      delta: float,
      one_sided: bool = True,
  ) -> float:
    """Computes the epsilon bound for a given threshold strategy."""
    match threshold_strategy:
      case Explicit(threshold):
        eps = audit_all_thresholds_method(
            self, significance, delta, one_sided, threshold
        )
      case Split(threshold_estimation_frac=p, seed=seed):
        rng = np.random.default_rng(seed)
        in_scores_1, in_scores_2 = _random_partition(
            self._in_canary_scores, rng, p
        )
        out_scores_1, out_scores_2 = _random_partition(
            self._out_canary_scores, rng, p
        )
        auditor_1 = CanaryScoreAuditor(in_scores_1, out_scores_1)
        _, threshold = audit_all_thresholds_method(
            auditor_1, significance, delta, one_sided, None
        )
        auditor_2 = CanaryScoreAuditor(in_scores_2, out_scores_2)
        eps = audit_all_thresholds_method(
            auditor_2, significance, delta, one_sided, threshold
        )
      case Bonferroni():
        significance_bonferroni = significance / len(self._fn_counts)
        eps, _ = audit_all_thresholds_method(
            self, significance_bonferroni, delta, one_sided, None
        )
      case _:
        raise ValueError(
            f'Unsupported threshold strategy: {threshold_strategy}.'
        )

    return eps
