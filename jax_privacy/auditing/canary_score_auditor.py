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

from collections.abc import Sequence
from concurrent import futures
import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats


_norm = scipy.stats.norm


@dataclasses.dataclass(frozen=True)
class BootstrapParams:
  """Parameters for bootstrapping.

  Attributes:
    num_samples: The number of times to resample the dataset.
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
  ) -> 'BootstrapParams':
    """Creates a BootstrapParams object for a confidence interval.

    Args:
      num_samples: The number of times to resample the dataset.
      confidence: The desired confidence level.
      seed: The seed for the random number generator.

    Returns:
      A BootstrapParams object.
    """
    if not 0 < confidence < 1:
      raise ValueError(f'confidence must be in (0, 1), got {confidence}.')
    alpha = 1 - confidence
    quantiles = (alpha / 2, 1 - alpha / 2)
    return cls(num_samples=num_samples, quantiles=quantiles, seed=seed)


def _log_sub(x, y):
  # Stable computation of log(exp(x) - exp(y)).
  if np.any(y > x):
    raise ValueError(f'y must be less than or equal to x, got y={y} and x={x}.')
  with np.errstate(divide='ignore'):  # Okay to return -np.inf if x == y.
    return x + np.log1p(-np.exp(y - x))


def _clopper_pearson_upper(
    k: int | np.ndarray, n: int, alpha: float
) -> float | np.ndarray:
  """Computes Clopper-Pearson one-sided upper binomial confidence interval.

  Args:
    k: The number of successes.
    n: The number of trials.
    alpha: Allowed probability of failure (one minus confidence).

  Returns:
    A value p such that the probability of observing k or fewer successes out of
    n Bernoilli(p) trials is approximately alpha.
  """
  return scipy.stats.beta.ppf(1 - alpha, k + 1, n - k)


@jax.jit
def _signed_area(a, b, c):
  """Computes (twice) the signed area of the triangle formed by three points."""
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@jax.jit
def _pareto_frontier_jax(points: jnp.ndarray) -> tuple[jnp.ndarray, jax.Array]:
  """JAX implementation of _pareto_frontier."""

  # Use the simple linear-time algorithm of Graham & Yao (1983).

  def scan_fn(carry, point):
    hull, n = carry

    def cond_fn(n):
      return (n > 1) & (_signed_area(hull[n - 2], hull[n - 1], point) >= 0)

    def body_fn(n):
      return n - 1

    # Pop points from the end of the hull until the current point makes a
    # clockwise turn from the last two.
    n = jax.lax.while_loop(cond_fn, body_fn, n)

    # Add the current point.
    hull = hull.at[n].set(point)
    return (hull, n + 1), None  # We don't need the scan to output anything.

  n = 2  # Number of points in the hull.
  hull = jnp.empty_like(points)
  hull = hull.at[:n].set(points[:n])

  (hull, n), _ = jax.lax.scan(scan_fn, (hull, n), points[n:])
  return hull, n  # pytype: disable=bad-return-type  # lax-types


def _pareto_frontier(points: np.ndarray) -> np.ndarray:
  """Computes the pareto frontier of a piecewise linear function.

  Given a piecewise linear function defined by a sequence of points, computes
  the set of points that are not weakly linearly dominated by any pair of outer
  points. That is, given a list of points (sorted by x coordinate), retain only
  points (x_i, y_i) for which there do not exist j < i and k > i and number a
  with 0 <= a <= 1 such that x_i = (1-a)x_j + ax_k and y_i <= (1-a)y_j + ay_k.

  Args:
    points: An array of shape (N, 2) with N >= 2 containing the vertices
      defining the segments of the piecewise linear function, sorted by x
      coordinate.

  Returns:
    A numpy array of shape (M, 2) with 2 <= M <= N, containing the subset of
    vertices_np on the pareto frontier.
  """
  if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
    raise ValueError(f'Expected at least two 2D points, got {points.shape}.')

  if not np.all(points[:-1, 0] <= points[1:, 0]):
    raise ValueError('Expected points to be sorted by x-coordinate.')

  hull, n = _pareto_frontier_jax(jnp.array(points))
  return np.array(hull[:n])


def _get_tn_fn_counts(
    in_canary_scores: Sequence[float],
    out_canary_scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
  """Computes true negative and false negative counts at each threshold.

  Args:
    in_canary_scores: Attack scores of held-in canaries.
    out_canary_scores: Attack scores of held-out canaries.

  Returns:
    A tuple (true negative counts, false negative counts) for each threshold.
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
  counts = _pareto_frontier(counts)
  fn_counts, tn_counts = np.unstack(counts, axis=1)

  return tn_counts, fn_counts


def _epsilon_raw_counts_helper(
    true_counts: np.ndarray,
    false_counts: np.ndarray,
    min_count: int,
    delta: float,
) -> float:
  """Estimates epsilon given true and false counts at each threshold."""
  n_true = np.max(true_counts)
  n_false = np.max(false_counts)

  pos_counts = np.stack([true_counts, false_counts], axis=1)
  pos_counts = pos_counts[np.min(pos_counts, axis=1) >= min_count]
  pos_rates = pos_counts / [n_true, n_false]

  if not pos_rates.size:
    # If no thresholds have required count, eps is zero.
    return 0

  # We want to ignore invalid values in the log here. If (tpr - delta) is
  # less or equal to zero, the bound is invalid. Let it return np.nan or -np.inf
  # and we will filter it with np.nanmax.
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.nanmax(np.log(pos_rates[:, 0] - delta) - np.log(pos_rates[:, 1]))


class CanaryScoreAuditor:
  """Class for auditing privacy based on attack scores."""

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
    self._n_in = self._in_canary_scores.size
    if self._n_in == 0:
      raise ValueError('in_canary_scores must be non-empty.')
    self._n_out = self._out_canary_scores.size
    if self._n_out == 0:
      raise ValueError('out_canary_scores must be non-empty.')

    self._tn_counts, self._fn_counts = _get_tn_fn_counts(
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

  def epsilon_lower_bound(
      self,
      alpha: float,
      delta: float = 0,
      one_sided: bool = True,
  ) -> float:
    """Finds epsilon lower bound from scores of held-in/held-out canaries.

    Args:
      alpha: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta.
      one_sided: Whether to use only TPR/FPR (vs. max of TPR/FPR and TNR/FNR).

    Returns:
      Optimal epsilon lower bound.
    """
    if not 0 < alpha < 0.5:
      raise ValueError(f'alpha must be in (0, 0.5), got {alpha}.')
    if not 0 <= delta <= 1:
      raise ValueError(f'delta must be in [0, 1], got {delta}.')

    # Use half of alpha for each so that the union bound is equal to alpha.
    fnr_ubs = _clopper_pearson_upper(self._fn_counts, self._n_in, alpha / 2)
    tpr_lbs = 1 - fnr_ubs
    fp_counts = self._n_out - self._tn_counts
    fpr_ubs = _clopper_pearson_upper(fp_counts, self._n_out, alpha / 2)

    # We want to ignore invalid values in the log here. If (tpr - delta) is less
    # or equal to zero, the bound is invalid. Let it return np.nan or -np.inf
    # and we will filter it with np.nanmax.
    with np.errstate(divide='ignore', invalid='ignore'):
      bound = np.nanmax(np.log(tpr_lbs - delta) - np.log(fpr_ubs))
      if not one_sided:
        tnr_lbs = 1 - fpr_ubs
        bound = max(bound, np.nanmax(np.log(tnr_lbs - delta) - np.log(fnr_ubs)))

    return max(bound, 0)

  def epsilon_raw_counts(
      self,
      min_count: int = 50,
      delta: float = 0,
      one_sided: bool = True,
      *,
      bootstrap_params: BootstrapParams | None = None,
  ) -> float | np.ndarray:
    """Estimates epsilon from raw count statistics of seen/unseen canaries.

    `min_count` is the minimum number of TP/FP (TN/FN) required to consider a
    threshold. If `min_count` is too high relative to the number of canaries,
    the estimate will be biased towards zero. If it is too low, the estimate
    will have high variance.

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

    tp_counts = self._n_in - self._fn_counts
    fp_counts = self._n_out - self._tn_counts
    eps = _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count, delta)

    if not one_sided:
      eps = max(
          eps,
          _epsilon_raw_counts_helper(
              self._tn_counts, self._fn_counts, min_count, delta
          ),
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
    # This implementation assumes that _tn_counts and _fn_counts are sorted in
    # increasing order, and have been filtered to contain only thresholds on the
    # boundary of the convex hull.
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

    target_tn_count = self._n_out * (1 - fpr)

    # Find the first threshold greater or equal to target_tn_count. We will
    # interpolate between the previous one and this one.
    threshold = np.searchsorted(self._tn_counts, target_tn_count)

    # If threshold is 0 (i.e., if target_tn_count is 0), we interpolate between
    # the first two thresholds. Because we have found the convex hull of FN/TN
    # counts, the first TN count is always 0 and the second is positive.
    threshold = np.maximum(threshold, 1)

    tn_left = self._tn_counts[threshold - 1]
    tn_right = self._tn_counts[threshold]
    p = (target_tn_count - tn_left) / (tn_right - tn_left)

    fn_left = self._fn_counts[threshold - 1]
    fn_right = self._fn_counts[threshold]
    fnr = ((1 - p) * fn_left + p * fn_right) / self._n_in

    return 1 - fnr

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
    # as a function of FNR, which is equivalent. Cast to float64 to prevent
    # overflow.
    tn_counts_float = self._tn_counts.astype(np.float64)
    fn_counts_float = self._fn_counts.astype(np.float64)
    unnorm = np.dot(
        tn_counts_float[:-1] + tn_counts_float[1:],
        fn_counts_float[1:] - fn_counts_float[:-1],
    )
    return 0.5 * unnorm / (self._n_in * self._n_out)

  def epsilon_from_gdp(
      self,
      alpha: float,
      delta: float,
      eps_tol: float = 1e-6,
  ) -> float:
    """Calculates the an estimate for epsilon with GDP.

    This is the method used in https://arxiv.org/pdf/2302.07956 and described in
    https://arxiv.org/pdf/2406.04827.

    Args:
      alpha: Allowed probability of failure (one minus confidence).
      delta: Approximate DP delta. Must be in (0, 1].
      eps_tol: The tolerance for epsilon (the privacy parameter). Defaults to
        1e-6.

    Returns:
        The estimated epsilon.
    """
    if not 0 < alpha < 0.5:
      raise ValueError(f'alpha must be in (0, 0.5), got {alpha}.')
    if not 0 < delta <= 1:
      raise ValueError(f'delta must be in (0, 1], got {delta}.')
    if eps_tol <= 0:
      raise ValueError(f'eps_tol must be positive, got {eps_tol}.')

    fnr_ubs = _clopper_pearson_upper(self._fn_counts, self._n_in, alpha / 2)
    fp_counts = self._n_out - self._tn_counts
    fpr_ubs = _clopper_pearson_upper(fp_counts, self._n_out, alpha / 2)

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

    eps = scipy.optimize.root_scalar(
        delta_gap,
        bracket=[eps_lb, eps_ub],
        method='brentq',
        xtol=eps_tol,
    ).root

    return eps
