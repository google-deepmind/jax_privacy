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

"""Tests for the Monte Carlo privacy-loss histogram accounting."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from research.monte_carlo import batch_selection
from research.monte_carlo import histogram
import numpy as np

jax.config.update('jax_enable_x64', True)

_C_COL = np.array([1.0, 0.5, 0.25], dtype=np.float64)
_SIGMA = 1.0

_STRATEGIES = {
    'balanced_min_sep': batch_selection.BalancedMinSep(
        min_sep=3, num_participations=2, iterations=12
    ),
    'balanced_min_sep_rotated': batch_selection.BalancedMinSep(
        min_sep=3, num_participations=2, iterations=12, random_rotation=True
    ),
    'nested_random_allocation': batch_selection.NestedRandomAllocation(
        cycle_length=3, num_participations=2, iterations=12
    ),
    'subsampled_balanced_min_sep': batch_selection.BalancedMinSep(
        min_sep=3, num_participations=2, iterations=12, sampling_prob=0.5
    ),
    'subsampled_nested_random_allocation': (
        batch_selection.NestedRandomAllocation(
            cycle_length=3,
            num_participations=2,
            iterations=12,
            sampling_prob=0.5,
        )
    ),
}


def _hist_mean_std(counts, grid_lo, grid_step):
  """Mean/std of the losses implied by a rounded-up histogram (no overflow)."""
  in_range = counts[:-1].astype(np.float64)  # Drop the +inf overflow bin.
  centers = grid_lo + np.arange(len(in_range)) * grid_step
  mean = np.average(centers, weights=in_range)
  var = np.average((centers - mean) ** 2, weights=in_range)
  return mean, np.sqrt(var)


def _gaussian_reduction_histogram(sigma):
  """Histogram for a degenerate strategy that collapses to a Gaussian."""
  # One iteration, one participation, min_sep=1 and a length-1 band reduce the
  # mechanism to a plain Gaussian with sensitivity 1 and noise sigma, whose
  # privacy loss is analytically N(mu, 2*mu) with mu = 1/(2*sigma**2).
  strategy = batch_selection.BalancedMinSep(
      min_sep=1, num_participations=1, iterations=1
  )
  return histogram.privacy_loss_histogram(
      strategy,
      np.array([1.0]),
      sigma,
      jax.random.PRNGKey(0),
      num_samples=200_000,
      grid_lo=-15.0,
      grid_hi=15.0,
      grid_step=0.02,
      microbatch_size=2000,
  )


class HistogramTest(parameterized.TestCase):

  def test_reduces_to_gaussian_mechanism(self):
    # The recovered loss distribution should match the closed-form Gaussian PLD,
    # N(mu, 2*mu) with mu = 1/(2*sigma**2), in both directions -- a check of the
    # whole pipeline with no reliance on a re-implementation of the scorer.
    sigma = 1.0
    hist = _gaussian_reduction_histogram(sigma)
    mu = 1.0 / (2.0 * sigma**2)
    std = np.sqrt(2.0 * mu)
    self.assertEqual(hist.overflow_fraction, 0.0)
    for counts in (hist.counts_pos, hist.counts_neg):
      mean, sd = _hist_mean_std(counts, hist.grid_lo, hist.grid_step)
      # Rounding up biases the mean up by <= grid_step; MC error is tiny at 2e5.
      self.assertAlmostEqual(mean, mu, delta=0.05)
      self.assertAlmostEqual(sd, std, delta=0.05)

  def test_narrow_grid_routes_mass_to_overflow(self):
    strategy = _STRATEGIES['balanced_min_sep']
    num_samples = 256
    hist = histogram.privacy_loss_histogram(
        strategy,
        _C_COL,
        _SIGMA,
        jax.random.PRNGKey(1),
        num_samples=num_samples,
        grid_lo=-0.5,
        grid_hi=0.5,
        grid_step=0.1,
        microbatch_size=64,
    )
    # No sample is lost even when the grid is far too narrow.
    self.assertEqual(hist.num_samples, num_samples)
    self.assertGreater(hist.overflow_fraction, 0.0)

  def test_result_is_invariant_to_microbatch_size(self):
    # The loop-index RNG makes counts independent of how samples are batched,
    # for any microbatch_size that divides num_samples.
    strategy = _STRATEGIES['nested_random_allocation']
    kwargs = dict(num_samples=256, grid_lo=-30.0, grid_hi=30.0, grid_step=0.1)
    ref = histogram.privacy_loss_histogram(
        strategy,
        _C_COL,
        _SIGMA,
        jax.random.PRNGKey(7),
        microbatch_size=256,
        **kwargs,
    )
    for microbatch_size in (1, 64, 128, 256):
      other = histogram.privacy_loss_histogram(
          strategy,
          _C_COL,
          _SIGMA,
          jax.random.PRNGKey(7),
          microbatch_size=microbatch_size,
          **kwargs,
      )
      self.assertEqual(other.num_samples, 256)
      np.testing.assert_array_equal(other.counts_pos, ref.counts_pos)
      np.testing.assert_array_equal(other.counts_neg, ref.counts_neg)

  def test_rejects_nonpositive_microbatch_size(self):
    with self.assertRaises(ValueError):
      histogram.privacy_loss_histogram(
          _STRATEGIES['balanced_min_sep'],
          _C_COL,
          _SIGMA,
          jax.random.PRNGKey(9),
          num_samples=100,
          microbatch_size=0,
      )

  @parameterized.named_parameters(
      ('balanced_min_sep_rotated', 'balanced_min_sep_rotated'),
      ('subsampled_balanced_min_sep', 'subsampled_balanced_min_sep'),
      ('subsampled_nested_ra', 'subsampled_nested_random_allocation'),
  )
  def test_strategy_runs_successfully(self, strategy_key):
    strategy = _STRATEGIES[strategy_key]
    hist = histogram.privacy_loss_histogram(
        strategy,
        _C_COL,
        _SIGMA,
        jax.random.PRNGKey(42),
        num_samples=256,
        grid_lo=-10.0,
        grid_hi=10.0,
        grid_step=0.1,
        microbatch_size=64,
    )
    self.assertEqual(hist.num_samples, 256)
    self.assertEqual(int(hist.counts_pos.sum()), 256)
    self.assertEqual(int(hist.counts_neg.sum()), 256)


def _per_sample(index):
  """Deterministic two-histogram kernel: bins depend only on the index."""
  return index % 5, (2 * index + 1) % 5


class AccumulateBincountsTest(parameterized.TestCase):

  @parameterized.parameters(1, 8, 64, 128, 256)
  def test_matches_numpy_over_dividing_microbatch_sizes(self, microbatch_size):
    num_samples, num_bins = 256, 5
    pos, neg = histogram._accumulate_bincounts(
        _per_sample, num_samples, num_bins, microbatch_size=microbatch_size
    )
    idx = np.arange(num_samples)
    np.testing.assert_array_equal(pos, np.bincount(idx % 5, minlength=num_bins))
    np.testing.assert_array_equal(
        neg, np.bincount((2 * idx + 1) % 5, minlength=num_bins)
    )
    self.assertEqual(int(pos.sum()), num_samples)

  def test_supports_single_array_output(self):
    out = histogram._accumulate_bincounts(
        lambda i: i % 3, 12, 3, microbatch_size=4
    )
    np.testing.assert_array_equal(
        out, np.bincount(np.arange(12) % 3, minlength=3)
    )


if __name__ == '__main__':
  absltest.main()
