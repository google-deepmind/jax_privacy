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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy import auditing
import numpy as np
import scipy.special
import scipy.stats

_binom = scipy.stats.binom
_logistic = scipy.special.expit

_signed_area = auditing._signed_area

_rotations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
_inversions = [[0, 2, 1], [1, 0, 2], [2, 1, 0]]


def _one_shot_p_value_naive(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
  """Naive implementation of the one-shot p-value."""
  q = _logistic(eps)
  beta = _binom.sf(n_correct - 1, n_guess, q)

  alpha = 0
  alpha_sum = 0
  for i in range(1, n_correct + 1):
    alpha_sum += _binom.pmf(n_correct - i, n_guess, q)
    if alpha_sum > i * alpha:
      alpha = alpha_sum / i

  return min(beta + alpha * delta * 2 * m, 1)


class CanaryScoreAuditorTest(parameterized.TestCase):

  def test_bootstrap_params_empty_quantiles(self):
    with self.assertRaisesRegex(ValueError, 'quantiles cannot be empty'):
      auditing.BootstrapParams(quantiles=[])

  @parameterized.named_parameters(('zero', (0, 0.5)), ('one', (0.5, 1)))
  def test_bootstrap_params_quantiles_out_of_range(self, quantiles):
    with self.assertRaisesRegex(ValueError, 'quantiles must be in'):
      auditing.BootstrapParams(quantiles=quantiles)

  def test_bootstrap_params_confidence_interval_illegal_confidence(self):
    for confidence in [-1, 0, 1, 2]:
      with self.assertRaisesRegex(ValueError, 'confidence must be in'):
        auditing.BootstrapParams.confidence_interval(
            confidence=confidence
        )

  @parameterized.named_parameters(
      ('95_percent', 0.95, [0.025, 0.975]),
      ('99_percent', 0.99, [0.005, 0.995]),
  )
  def test_bootstrap_params_confidence_interval_correct_quantiles(
      self,
      confidence,
      expected_quantiles,
  ):
    params = auditing.BootstrapParams.confidence_interval(
        confidence=confidence
    )
    np.testing.assert_almost_equal(params.quantiles, expected_quantiles)

  @parameterized.product(
      n=[10, 100, 1000],
      k_frac=[0.3, 0.4, 0.5, 0.8, 0.9],
      alpha=[0.1, 0.03, 0.01],
  )
  def test_clopper_pearson_upper(self, n, k_frac, alpha):
    trials = 1_000_000
    k = k_frac * n
    p = auditing._clopper_pearson_upper(k, n, alpha)
    rng = np.random.default_rng(seed=0xBAD5EED)
    np.testing.assert_allclose(
        np.mean(rng.binomial(n, p, size=trials) <= k),
        alpha,
        rtol=0.05,
    )

  @parameterized.parameters(
      ([[0, 0], [1, 0], [1, 1]], 1),
      ([[0, 0], [2, 0], [1, 2]], 4),
      ([[1, 0], [2, 3], [0, 2]], 5),
      ([[0, 0], [1, 1], [2, 2]], 0),
      ([[-4, 3], [6, -2], [-4, 3]], 0),
  )
  def test_signed_area(self, points, expected):
    points = np.array(points)
    for perm in _rotations:
      self.assertEqual(_signed_area(*points[perm]), expected)
    for perm in _inversions:
      self.assertEqual(_signed_area(*points[perm]), -expected)

  @parameterized.parameters(
      ((),),
      ((1,),),
      ((1, 2),),
      ((2, 1),),
      ((2, 3),),
      ((1, 2, 3),),
  )
  def test_pareto_frontier_bad_shape(self, shape):
    points = np.zeros(shape, dtype=np.int64)
    with self.assertRaisesRegex(
        ValueError, 'Expected at least two 2D points'
    ):
      auditing._pareto_frontier(points)

  def test_pareto_frontier_unsorted(self):
    points = np.array([[1, 0], [0, 1]])
    with self.assertRaisesRegex(
        ValueError, 'Expected points to be sorted'
    ):
      auditing._pareto_frontier(points)

  def test_pareto_frontier_two_points(self):
    points = np.array([[0, 0], [1, 1]])
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, points)

  def test_pareto_frontier_linear(self):
    n = 100
    points = np.stack([range(n), range(n)], axis=1)
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, points[[0, -1]])

  def test_pareto_frontier_simple_1(self):
    points = np.array([[0, 0], [0, 2], [3, 2], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    np.testing.assert_equal(frontier, points[[0, 1, 3, 4]])

  def test_pareto_frontier_simple_2(self):
    points = np.array([[0, 0], [0, 2], [2, 2], [2, 3], [3, 3], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    # Should not contain [3, 3], which is dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(frontier, points[[0, 1, 5, 6]])

  def test_pareto_frontier_simple_3(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 4], [3, 4], [3, 5], [5, 5]])
    frontier = auditing._pareto_frontier(points)
    # Should contain [1, 4], which is not dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(frontier, points[[0, 1, 3, 5, 6]])

  def test_pareto_frontier_simple_4(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [4, 4]])
    frontier = auditing._pareto_frontier(points)
    # Should not contain [1, 3], which is a combination of [0, 2] and [2, 4].
    np.testing.assert_equal(frontier, points[[0, 1, 5, 6]])

  @parameterized.named_parameters(
      ('increasing', np.sin, np.pi / 2),
      ('decreasing', np.cos, np.pi / 2),
      ('increasing_and_decreasing', np.sin, np.pi),
  )
  def test_pareto_frontier_convex(self, fn, bound):
    xs = np.linspace(0, bound, 100)
    points = np.stack([xs, fn(xs)], axis=1)
    frontier = auditing._pareto_frontier(points)
    # On a convex function, the frontier should be the same as the points.
    np.testing.assert_almost_equal(frontier, points)

  @parameterized.named_parameters(
      ('increasing', lambda x: -np.cos(x), np.pi / 2),
      ('decreasing', lambda x: -np.sin(x), np.pi / 2),
      ('decreasing_and_increasing', lambda x: -np.sin(x), np.pi),
  )
  def test_pareto_frontier_concave(self, fn, bound):
    xs = np.linspace(0, bound, 100)
    points = np.stack([xs, fn(xs)], axis=1)
    frontier = auditing._pareto_frontier(points)
    # On a concave function, the frontier should be the first and last points.
    np.testing.assert_almost_equal(frontier, points[[0, -1]])

  @parameterized.parameters(range(10))
  def test_pareto_frontier_random(self, seed):
    n = 80
    rng = np.random.default_rng(seed=0xBAD5EED + seed)
    xs = np.linspace(0, np.pi, n)
    ys = np.sin(xs) + rng.normal(scale=0.1, size=n)
    points = np.stack([xs, ys], axis=1)
    frontier = auditing._pareto_frontier(points)

    # Compare to simple cubic time algorithm.
    is_frontier = [True] * n
    for i, j, k in itertools.combinations(range(n), 3):
      if _signed_area(points[i], points[j], points[k]) >= 0:
        is_frontier[j] = False
    expected_frontier = points[is_frontier]

    np.testing.assert_almost_equal(frontier, expected_frontier)

  def test_get_tn_fn_counts(self):
    in_canary_scores = np.arange(4) + 0.5
    np.random.shuffle(in_canary_scores)
    out_canary_scores = np.arange(4)
    np.random.shuffle(out_canary_scores)
    tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 1, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 0, 3, 4])

  def test_get_tn_fn_counts_ties_convex(self):
    in_canary_scores = [0, 1, 1, 2]
    out_canary_scores = [0, 0, 1, 1]
    tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 2, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 1, 3, 4])

  def test_get_tn_fn_counts_ties_nonconvex(self):
    in_canary_scores = [0, 0, 1, 2]
    out_canary_scores = [0, 1, 1, 1]
    tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 3, 4])

  def test_get_tn_fn_counts_zeros(self):
    in_canary_scores = np.zeros(4)
    out_canary_scores = np.zeros(4)
    tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 4])
    np.testing.assert_equal(fn_counts, [0, 4])

  @parameterized.product(
      one_sided=(True, False),
      mu=(0.3, 1.0, 1.5),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_lower_bound_tight(self, one_sided, mu, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)

    # Large alpha and delta and lots of samples gets us close to the true eps.
    alpha = 0.2
    delta = 0.1
    in_samples = 2_000_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(mu, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps_lb = auditor.epsilon_lower_bound(alpha, delta, one_sided)
    true_eps = dp_accounting.get_epsilon_gaussian(1 / mu, delta)
    np.testing.assert_array_less(eps_lb, true_eps)
    np.testing.assert_allclose(eps_lb, true_eps, rtol=0.2)

  @parameterized.product(
      min_count=(0, 1, 50),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_raw_counts_helper_accurate_large_delta(
      self, min_count, out_samples_ratio
  ):
    rng = np.random.default_rng(seed=0xBAD5EED)

    delta = 1e-2
    in_samples = 200_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(1, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    tn_counts, fn_counts = auditing._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    eps = auditing._epsilon_raw_counts_helper(
        tn_counts, fn_counts, min_count, delta
    )
    true_eps = dp_accounting.get_epsilon_gaussian(1.0, delta)
    np.testing.assert_allclose(eps, true_eps, rtol=1e-1)

  @parameterized.product(
      min_count=(1, 3),
      n_neg=(10, 13),
      n_pos=(9, 12),
  )
  def test_epsilon_raw_counts_helper_worst_case(self, min_count, n_neg, n_pos):
    # Tests that the epsilon uses the minimum allowed FPR for a perfect
    # classifier.
    tn_counts = np.array([0, n_neg, n_neg])
    fn_counts = np.array([0, 0, n_pos])
    epsilon = auditing._epsilon_raw_counts_helper(
        tn_counts, fn_counts, min_count, delta=0
    )
    np.testing.assert_allclose(epsilon, np.log(n_neg / min_count))

  @parameterized.product(
      min_count=[0, 1, 2],
      delta=[0, 1 / 8, 1 / 4, 3 / 8, 1 / 2],
  )
  def test_epsilon_raw_counts_helper_nonzero_delta(self, min_count, delta):
    tn_counts = np.array([0, 4, 8, 8])
    fn_counts = np.array([0, 2, 6, 8])
    epsilon = auditing._epsilon_raw_counts_helper(
        tn_counts, fn_counts, min_count, delta
    )
    if min_count == 0:
      expected_slopes = {0: np.inf, 1 / 8: np.inf}
    elif min_count == 1:
      expected_slopes = {0: 3, 1 / 8: 2}
    else:  # min_count == 2
      expected_slopes = {0: 2, 1 / 8: 3 / 2}
    expected_slopes.update({1 / 4: 1, 3 / 8: 3 / 4, 1 / 2: 1 / 2})
    np.testing.assert_allclose(epsilon, max(np.log(expected_slopes[delta]), 0))

  @parameterized.product(
      one_sided=(True, False),
      delta=(0, 0.1),
  )
  def test_epsilon_raw_counts_zero_if_no_min_counts(self, one_sided, delta):
    in_canary_scores = np.arange(10) + 0.5
    out_canary_scores = np.arange(10)
    min_count = 12
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps = auditor.epsilon_raw_counts(min_count, delta, one_sided)
    self.assertEqual(eps, 0.0)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      vectorized=(True, False),
  )
  def test_tpr_at_given_fpr_worst_case(self, out_samples_ratio, vectorized):
    """Test that TPR equals FPR for a worst-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 0.0)
    out_canary_scores = np.full(out_samples, 1.0)
    auditor = auditing.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    np.testing.assert_allclose(tprs, fprs)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
      vectorized=(True, False),
  )
  def test_tpr_at_given_fpr_best_case(self, out_samples_ratio, vectorized):
    """Test that TPR at FPR is 1.0 for a best-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 1.0)
    out_canary_scores = np.full(out_samples, 0.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    np.testing.assert_equal(tprs, np.ones_like(fprs))

  def test_tpr_at_given_fpr_simple(self):
    in_canary_scores = [1, 3]
    out_canary_scores = [0, 2]
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = [0, 0.25, 0.5, 0.75, 1.0]
    tprs = auditor.tpr_at_given_fpr(fprs)
    expected_tprs = [0.5, 0.75, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(tprs, expected_tprs)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_tpr_at_given_fpr_random(self, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_samples = 100_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.uniform(0, 1, in_samples)
    out_canary_scores = rng.uniform(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    fprs = np.linspace(0, 1, 10)
    tprs = auditor.tpr_at_given_fpr(fprs)
    np.testing.assert_allclose(tprs, fprs, rtol=0.05, atol=1e-3)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_worst_case(self, out_samples_ratio):
    """Test that auroc is 0.5 for a worst-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 0.0)
    out_canary_scores = np.full(out_samples, 1.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 0.5)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_best_case(self, out_samples_ratio):
    """Test that auroc is 1.0 for a best-case dataset."""
    in_samples = 30
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = np.full(in_samples, 1.0)
    out_canary_scores = np.full(out_samples, 0.0)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 1.0)

  def test_attack_auroc_simple(self):
    in_canary_scores = [0, 4, 6, 7]
    out_canary_scores = [1, 2, 3, 5]

    # All FP/TP points: (0,0) (0,1) (0,2) (1,2) (1,3) (2,3) (2,4) (3,4) (4,4)
    # FP/TP frontier points: (0,0) (0,2) (1,3) (4,4)
    # Area using trapezoids: (0 + 2.5 + 10.5) / 16 = 13 / 16

    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    self.assertAlmostEqual(auditor.attack_auroc(), 13 / 16)

  @parameterized.product(
      in_samples=(10_000, 100_000),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_random(self, in_samples, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.uniform(0, 1, in_samples)
    out_canary_scores = rng.uniform(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    np.testing.assert_allclose(auditor.attack_auroc(), 0.5, rtol=0.05)

  @parameterized.product(
      mu=(0.1, 0.3, 1.0, 3.0),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_from_gdp_tight(self, mu, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)

    # Large alpha and delta and lots of samples gets us close to the true eps.
    alpha = 0.1
    delta = 5e-2
    in_samples = 500_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(mu, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    eps = auditor.epsilon_from_gdp(alpha, delta)
    true_eps = dp_accounting.get_epsilon_gaussian(1 / mu, delta)
    np.testing.assert_allclose(eps, true_eps, rtol=0.05)

  @parameterized.named_parameters(
      ('left', 0.025),
      ('right', 0.975),
      ('two_sided', (0.025, 0.975)),
      ('two_sided_with_median', (0.025, 0.5, 0.975)),
  )
  def test_bootstrap(self, quantiles):
    rng = np.random.default_rng(seed=0xBAD5EED)
    n = 5000

    # Compute interval for mean of scores, which we can also get exactly with
    # the central limit theorem. Use any crazy distribution for the data.
    in_canary_scores = rng.normal(np.e, np.pi, n // 2)
    out_canary_scores = rng.uniform(0, 1, n // 2)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)

    def mean_score(a: auditing.CanaryScoreAuditor):
      return np.mean([a._in_canary_scores, a._out_canary_scores])

    bootstrap_params = auditing.BootstrapParams(
        quantiles=quantiles,
        seed=0xBAD5EED,
    )
    interval = auditor._bootstrap(mean_score, bootstrap_params)
    mu_hat = np.mean([in_canary_scores, out_canary_scores])
    sigma_hat = np.std([in_canary_scores, out_canary_scores])
    expected_interval = auditing._norm.ppf(
        q=quantiles,
        loc=mu_hat,
        scale=sigma_hat / np.sqrt(n),
    )
    np.testing.assert_allclose(interval, expected_interval, rtol=0.01)

  def test_tpr_at_given_fpr_bootstrap_raises_non_scalar_fpr(self):
    auditor = auditing.CanaryScoreAuditor(
        in_canary_scores=[1, 2, 3], out_canary_scores=[0, 1, 2]
    )
    bootstrap_params = auditing.BootstrapParams()
    fpr = [0.1, 0.2]
    with self.assertRaisesRegex(
        ValueError, 'fpr must be a scalar for bootstrap'
    ):
      auditor.tpr_at_given_fpr(fpr, bootstrap_params=bootstrap_params)

  @parameterized.named_parameters(
      ('epsilon_raw_counts', 'epsilon_raw_counts', ()),
      ('tpr_at_given_fpr', 'tpr_at_given_fpr', (0.1,)),
      ('attack_auroc', 'attack_auroc', ()),
  )
  def test_auditing_metric_bootstrap(self, metric_fn_name, args):
    # Test that bootstrapped metrics run and return basically reasonable values.
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_canary_scores = rng.normal(size=356)
    out_canary_scores = rng.normal(size=432)
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    bootstrap_params = auditing.BootstrapParams(seed=0xBAD5EED)
    metric_fn = getattr(auditor, metric_fn_name)
    value = metric_fn(*args)
    interval = metric_fn(*args, bootstrap_params=bootstrap_params)
    self.assertLen(interval, 2)
    self.assertBetween(value, *interval)

  @parameterized.named_parameters(
      ('significance_zero', {'significance': 0.0}, 'significance'),
      ('significance_gt_one', {'significance': 1.1}, 'significance'),
      ('delta_negative', {'delta': -0.1}, 'delta'),
      ('delta_gt_one', {'delta': 1.1}, 'delta'),
      ('one_sided_false', {'one_sided': False}, 'one_sided must be True'),
  )
  def test_epsilon_one_shot_raises_invalid_args(
      self, override_args, error_regex
  ):
    args = {'significance': 0.1, 'delta': 0.1, 'one_sided': True}
    args.update(override_args)
    auditor = auditing.CanaryScoreAuditor(
        in_canary_scores=[1, 2, 3], out_canary_scores=[0, 1, 2]
    )
    with self.assertRaisesRegex(ValueError, error_regex):
      auditor.epsilon_one_shot(**args)

  @parameterized.product(
      m=(10, 100, 1000),
      n_guess=(1, 10, 100),
      n_wrong=(0, 1, 10, 100),
      eps=(0, 0.5, 1.0),
      delta=(0, 1e-6, 1e-3, 1e-1, 1),
  )
  def test_one_shot_p_value(self, m, n_guess, n_wrong, eps, delta):
    if not n_wrong <= n_guess <= m:
      return
    n_correct = n_guess - n_wrong
    p = auditing._one_shot_p_value(m, n_guess, n_correct, eps, delta)
    naive_p = _one_shot_p_value_naive(m, n_guess, n_correct, eps, delta)
    self.assertAlmostEqual(p, naive_p)

  @parameterized.named_parameters(
      ('zero', 0, 0),
      ('one', 1, 1.13346),
      ('two', 2, 2.311945),
      ('four', 4, 4.395568),
  )
  def test_epsilon_one_shot_close(self, shift, expected_eps):
    n = 10000
    # Scores deterministically distributed like Normal(0, 1).
    out_canary_scores = scipy.stats.norm.ppf(
        np.linspace(0, 1, n, endpoint=False) + 1 / (2 * n)
    )
    in_canary_scores = out_canary_scores + shift
    auditor = auditing.CanaryScoreAuditor(in_canary_scores, out_canary_scores)
    significance = 0.05
    delta = 1e-6
    eps = auditor.epsilon_one_shot(significance, delta)
    np.testing.assert_allclose(eps, expected_eps, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
