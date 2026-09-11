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

"""Tests for privacy-loss scoring, checked against brute-force enumeration."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from research.monte_carlo import batch_selection
from research.monte_carlo import scoring
import numpy as np
import scipy.special

jax.config.update('jax_enable_x64', True)


def _llrs_np(c_col, sample, sigma):
  """Per-iteration Gaussian LLRs, computed directly for a single sample."""
  t = sample.shape[0]
  b = c_col.shape[0]
  dots = np.array([np.dot(c_col[: t - i], sample[i : i + b]) for i in range(t)])
  sq = np.array([np.sum(c_col[: min(b, t - i)] ** 2) for i in range(t)])
  return (2.0 * dots - sq) / (2.0 * sigma**2)


def _marginalize_over_kept(llr, subset, q):
  """log E over kept subsets A of ``subset`` (each kept iid w.p. q, 0<q<1)."""
  # Enumerate every kept subset A explicitly, independent of the scorer's
  # closed form.
  terms = []
  for r in range(len(subset) + 1):
    for kept in itertools.combinations(subset, r):
      log_w = r * np.log(q) + (len(subset) - r) * np.log1p(-q)
      terms.append(log_w + sum(llr[j] for j in kept))
  return scipy.special.logsumexp(terms)


def _brute_force_balanced(c_col, sample, sigma, k, b, q=1.0):
  """log privacy loss by enumerating all min-sep K-subsets (uniform)."""
  t = sample.shape[0]
  llr = _llrs_np(c_col, sample, sigma)
  terms = []
  for subset in itertools.combinations(range(t), k):
    gaps = np.diff(subset)
    if k > 1 and np.any(gaps < b):
      continue
    if q < 1.0:
      terms.append(_marginalize_over_kept(llr, subset, q))
    else:
      terms.append(sum(llr[j] for j in subset))
  # Linear b-min-sep is uniform over the valid subsets.
  return scipy.special.logsumexp(terms) - np.log(len(terms))


def _brute_force_balanced_cyclic(c_col, sample, sigma, k, b, q=1.0):
  """log privacy loss over cyclic min-sep K-subsets (uniform, wrap-aware)."""
  t = sample.shape[0]
  llr = _llrs_np(c_col, sample, sigma)
  terms = []
  for subset in itertools.combinations(range(t), k):
    gaps = list(np.diff(subset)) + [subset[0] + t - subset[-1]]  # + wrap gap
    if np.any(np.array(gaps) < b):
      continue
    if q < 1.0:
      terms.append(_marginalize_over_kept(llr, subset, q))
    else:
      terms.append(sum(llr[j] for j in subset))
  return scipy.special.logsumexp(terms) - np.log(len(terms))


def _brute_force_nested(c_col, sample, sigma, k, b, q=1.0):
  """log privacy loss by enumerating K-subsets within each residue class."""
  t = sample.shape[0]
  llr = _llrs_np(c_col, sample, sigma)
  per_bin = []
  for i in range(b):
    candidates = list(range(i, t, b))
    subset_sums = []
    for subset in itertools.combinations(candidates, k):
      if q < 1.0:
        subset_sums.append(_marginalize_over_kept(llr, subset, q))
      else:
        subset_sums.append(sum(llr[j] for j in subset))
    # Uniform K-subset within the bin.
    per_bin.append(
        scipy.special.logsumexp(subset_sums) - np.log(len(subset_sums))
    )
  return scipy.special.logsumexp(per_bin) - np.log(b)


class BalancedMinSepScoringTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(t=8, k=2, b=2),
      dict(t=10, k=3, b=2),
      dict(t=9, k=2, b=3),
      dict(t=7, k=1, b=2),
      dict(t=6, k=2, b=3),
  )
  def test_matches_brute_force(self, t, k, b):
    rng = np.random.default_rng(t * 100 + k * 10 + b)
    c_col = rng.standard_normal(b)
    sigma = 1.3
    sample = rng.standard_normal(t)
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_balanced(c_col, sample, sigma, k, b)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(t=8, k=2, b=2, q=0.5),
      dict(t=10, k=3, b=2, q=0.3),
      dict(t=9, k=2, b=3, q=0.7),
      dict(t=6, k=2, b=3, q=0.5),
  )
  def test_subsampled_matches_brute_force(self, t, k, b, q):
    rng = np.random.default_rng(t * 100 + k * 10 + b + int(q * 100))
    c_col = rng.standard_normal(b)
    sigma = 1.3
    sample = rng.standard_normal(t)
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t, sampling_prob=q
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_balanced(c_col, sample, sigma, k, b, q)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(t=8, k=2, b=2),
      dict(t=10, k=3, b=2),
      dict(t=9, k=2, b=3),
      dict(t=7, k=1, b=2),
      dict(t=16, k=4, b=4),
      dict(t=12, k=2, b=4),
  )
  def test_random_rotation_matches_brute_force(self, t, k, b):
    rng = np.random.default_rng(t * 100 + k * 10 + b + 7)
    c_col = rng.standard_normal(b)
    sigma = 1.3
    sample = rng.standard_normal(t)
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t, random_rotation=True
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_balanced_cyclic(c_col, sample, sigma, k, b)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(t=8, k=2, b=2, q=0.5),
      dict(t=10, k=3, b=2, q=0.3),
      dict(t=9, k=2, b=3, q=0.7),
      dict(t=12, k=2, b=4, q=0.5),
  )
  def test_random_rotation_subsampled_matches_brute_force(self, t, k, b, q):
    rng = np.random.default_rng(t * 100 + k * 10 + b + int(q * 100) + 7)
    c_col = rng.standard_normal(b)
    sigma = 1.3
    sample = rng.standard_normal(t)
    strategy = batch_selection.BalancedMinSep(
        min_sep=b,
        num_participations=k,
        iterations=t,
        random_rotation=True,
        sampling_prob=q,
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_balanced_cyclic(c_col, sample, sigma, k, b, q)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class NestedRandomAllocationScoringTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(t=8, k=2, b=2),
      dict(t=9, k=2, b=3),
      dict(t=12, k=3, b=2),
      dict(t=6, k=1, b=2),
      dict(t=10, k=2, b=5),
  )
  def test_matches_brute_force(self, t, k, b):
    rng = np.random.default_rng(t * 100 + k * 10 + b + 1)
    c_col = rng.standard_normal(b)
    sigma = 0.9
    sample = rng.standard_normal(t)
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_nested(c_col, sample, sigma, k, b)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(t=8, k=2, b=2, q=0.5),
      dict(t=12, k=3, b=2, q=0.3),
      dict(t=10, k=2, b=5, q=0.7),
      # k == t // b: every candidate selected, so this is the cyclic-Poisson
      # reduction, checked against explicit enumeration of the kept set.
      dict(t=9, k=3, b=3, q=0.5),
  )
  def test_subsampled_matches_brute_force(self, t, k, b, q):
    rng = np.random.default_rng(t * 100 + k * 10 + b + int(q * 100) + 1)
    c_col = rng.standard_normal(b)
    sigma = 0.9
    sample = rng.standard_normal(t)
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t, sampling_prob=q
    )
    got = float(scoring.compute_privacy_loss(strategy, sample, sigma, c_col))
    expected = _brute_force_nested(c_col, sample, sigma, k, b, q)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class ScoringApiTest(parameterized.TestCase):

  def test_min_sep_one_matches_across_strategies(self):
    # With b == 1 both strategies are plain random allocation, so they agree.
    rng = np.random.default_rng(0)
    t, k = 6, 2
    c_col = np.array([1.0])
    sigma = 1.1
    sample = rng.standard_normal(t)
    balanced = batch_selection.BalancedMinSep(
        min_sep=1, num_participations=k, iterations=t
    )
    nested = batch_selection.NestedRandomAllocation(
        cycle_length=1, num_participations=k, iterations=t
    )
    np.testing.assert_allclose(
        float(scoring.compute_privacy_loss(balanced, sample, sigma, c_col)),
        float(scoring.compute_privacy_loss(nested, sample, sigma, c_col)),
        rtol=1e-6,
        atol=1e-6,
    )

  def test_jit_and_vmap_over_samples(self):
    rng = np.random.default_rng(3)
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=2, num_participations=2, iterations=8
    )
    c_col = np.array([1.0, 0.5])
    samples = rng.standard_normal((8, 5))  # (iterations, num_samples)
    fn = jax.jit(
        jax.vmap(
            lambda s: scoring.compute_privacy_loss(strategy, s, 1.0, c_col),
            in_axes=1,
        )
    )
    self.assertEqual(fn(samples).shape, (5,))

  def test_requires_1d_sample(self):
    strategy = batch_selection.BalancedMinSep(
        min_sep=2, num_participations=2, iterations=8
    )
    with self.assertRaises(ValueError):
      scoring.compute_privacy_loss(
          strategy, np.zeros((8, 1)), 1.0, np.array([1.0])
      )

  @parameterized.parameters('balanced', 'nested')
  def test_sampling_prob_one_is_exact_noop(self, kind):
    rng = np.random.default_rng(4)
    t, k, b = 10, 3, 2
    c_col = rng.standard_normal(b)
    sigma = 1.2
    sample = rng.standard_normal(t)
    if kind == 'balanced':
      make = lambda q: batch_selection.BalancedMinSep(
          min_sep=b, num_participations=k, iterations=t, sampling_prob=q
      )
    else:
      make = lambda q: batch_selection.NestedRandomAllocation(
          cycle_length=b, num_participations=k, iterations=t, sampling_prob=q
      )
    default = scoring.compute_privacy_loss(make(1.0), sample, sigma, c_col)
    # Recreate the strategy without passing sampling_prob (defaults to 1.0).
    np.testing.assert_array_equal(
        np.asarray(default),
        np.asarray(
            scoring.compute_privacy_loss(make(1.0), sample, sigma, c_col)
        ),
    )
    # A slightly subsampled variant must differ, confirming the branch is live.
    self.assertNotAlmostEqual(
        float(default),
        float(scoring.compute_privacy_loss(make(0.9), sample, sigma, c_col)),
    )


if __name__ == '__main__':
  absltest.main()
