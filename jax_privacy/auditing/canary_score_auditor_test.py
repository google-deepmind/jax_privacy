# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy.auditing import canary_score_auditor
import numpy as np


class CanaryScoreAuditorTest(parameterized.TestCase):

  @parameterized.product(
      n=[10, 100, 1000],
      k_frac=[0.3, 0.4, 0.5, 0.8, 0.9],
      alpha=[0.1, 0.03, 0.01],
  )
  def test_clopper_pearson_upper(self, n, k_frac, alpha):
    trials = 1_000_000
    k = k_frac * n
    p = canary_score_auditor._clopper_pearson_upper(k, n, alpha)
    rng = np.random.default_rng(seed=0xBAD5EED)
    np.testing.assert_allclose(
        np.mean(rng.binomial(n, p, size=trials) <= k),
        alpha,
        rtol=0.05,
    )

  def test_get_tn_fn_counts(self):
    in_canary_scores = np.arange(4) + 0.5
    np.random.shuffle(in_canary_scores)
    out_canary_scores = np.arange(4)
    np.random.shuffle(out_canary_scores)
    tn_counts, fn_counts = canary_score_auditor._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 1, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 0, 3, 4])

  def test_get_tn_fn_counts_ties_convex(self):
    in_canary_scores = [0, 1, 1, 2]
    out_canary_scores = [0, 0, 1, 1]
    tn_counts, fn_counts = canary_score_auditor._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 2, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 1, 3, 4])

  def test_get_tn_fn_counts_ties_nonconvex(self):
    in_canary_scores = [0, 0, 1, 2]
    out_canary_scores = [0, 1, 1, 1]
    tn_counts, fn_counts = canary_score_auditor._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 4, 4])
    np.testing.assert_equal(fn_counts, [0, 3, 4])

  def test_get_tn_fn_counts_zeros(self):
    in_canary_scores = np.zeros(4)
    out_canary_scores = np.zeros(4)
    tn_counts, fn_counts = canary_score_auditor._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    np.testing.assert_equal(tn_counts, [0, 4])
    np.testing.assert_equal(fn_counts, [0, 4])

  @parameterized.product(
      one_sided=(True, False),
      mu=(0.1, 0.3, 1.0, 1.5),
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_epsilon_lower_bound_tight(self, one_sided, mu, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)

    # Large alpha and delta and lots of samples gets us close to the true eps.
    alpha = 0.1
    delta = 5e-2
    in_samples = 200_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(mu, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    eps = auditor.epsilon_lower_bound(alpha, delta, one_sided)
    true_eps = (
        dp_accounting.pld.PLDAccountant()
        .compose(dp_accounting.GaussianDpEvent(1.0 / mu))
        .get_epsilon(delta)
    )
    np.testing.assert_allclose(eps, true_eps, rtol=0.05)

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
    tn_counts, fn_counts = canary_score_auditor._get_tn_fn_counts(
        in_canary_scores, out_canary_scores
    )
    eps = canary_score_auditor._epsilon_raw_counts_helper(
        tn_counts, fn_counts, min_count, delta
    )
    true_eps = (
        dp_accounting.pld.PLDAccountant()
        .compose(dp_accounting.GaussianDpEvent(1.0))
        .get_epsilon(delta)
    )
    np.testing.assert_allclose(eps, true_eps, rtol=1e-1)

  @parameterized.named_parameters(('one_sided', True), ('two_sided', False))
  def test_epsilon_raw_counts_zero_if_no_min_counts(self, one_sided):
    in_canary_scores = np.arange(10) + 0.5
    out_canary_scores = np.arange(10)
    delta = 1e-3
    min_count = 12
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    fprs = np.linspace(0, 1, 10)
    if vectorized:
      tprs = auditor.tpr_at_given_fpr(fprs)
    else:
      tprs = [auditor.tpr_at_given_fpr(fpr) for fpr in fprs]
    np.testing.assert_equal(tprs, np.ones_like(fprs))

  def test_tpr_at_given_fpr_simple(self):
    in_canary_scores = [1, 3]
    out_canary_scores = [0, 2]
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    self.assertAlmostEqual(auditor.attack_auroc(), 1.0)

  def test_attack_auroc_simple(self):
    in_canary_scores = [0, 4, 6, 7]
    out_canary_scores = [1, 2, 3, 5]

    # All FP/TP points: (0,0) (0,1) (0,2) (1,2) (1,3) (2,3) (2,4) (3,4) (4,4)
    # FP/TP hull points: (0,0) (0,2) (1,3) (4,4)
    # Area using trapezoids: (0 + 2.5 + 10.5) / 16 = 13 / 16

    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    self.assertAlmostEqual(auditor.attack_auroc(), 13 / 16)

  @parameterized.product(
      out_samples_ratio=(0.5, 1.0, 1.5),
  )
  def test_attack_auroc_random(self, out_samples_ratio):
    rng = np.random.default_rng(seed=0xBAD5EED)
    in_samples = 10_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.uniform(0, 1, in_samples)
    out_canary_scores = rng.uniform(0, 1, out_samples)
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
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
    in_samples = 100_000
    out_samples = int(in_samples * out_samples_ratio)
    in_canary_scores = rng.normal(mu, 1, in_samples)
    out_canary_scores = rng.normal(0, 1, out_samples)
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    eps = auditor.epsilon_from_gdp(alpha, delta)
    true_eps = (
        dp_accounting.pld.PLDAccountant()
        .compose(dp_accounting.GaussianDpEvent(1.0 / mu))
        .get_epsilon(delta)
    )
    np.testing.assert_allclose(eps, true_eps, rtol=0.05)

  def test_bootstrap(self):
    rng = np.random.default_rng(seed=0xBAD5EED)
    n = 10000

    # Compute interval for mean of scores, which we can also get exactly with
    # the central limit theorem. Use any crazy distribution for the data.
    in_canary_scores = rng.normal(np.e, np.pi, n // 2)
    out_canary_scores = rng.uniform(0, 1, n // 2)
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )

    def mean_score(a: canary_score_auditor.CanaryScoreAuditor):
      return np.mean([a._in_canary_scores, a._out_canary_scores])

    bootstrap_params = canary_score_auditor.BootstrapParams(seed=0xBAD5EED)
    interval = auditor._bootstrap(mean_score, bootstrap_params)
    mu_hat = np.mean([in_canary_scores, out_canary_scores])
    sigma_hat = np.std([in_canary_scores, out_canary_scores])
    expected_interval = canary_score_auditor._norm.interval(
        confidence=bootstrap_params.confidence,
        loc=mu_hat,
        scale=sigma_hat / np.sqrt(n),
    )
    np.testing.assert_allclose(interval, expected_interval, rtol=0.01)

  def test_tpr_at_given_fpr_bootstrap_raises_non_scalar_fpr(self):
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores=[1, 2, 3], out_canary_scores=[0, 1, 2]
    )
    bootstrap_params = canary_score_auditor.BootstrapParams()
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
    auditor = canary_score_auditor.CanaryScoreAuditor(
        in_canary_scores, out_canary_scores
    )
    bootstrap_params = canary_score_auditor.BootstrapParams(seed=0xBAD5EED)
    metric_fn = getattr(auditor, metric_fn_name)
    value = metric_fn(*args)
    interval = metric_fn(*args, bootstrap_params=bootstrap_params)
    self.assertLen(interval, 2)
    self.assertBetween(value, *interval)


if __name__ == '__main__':
  absltest.main()
