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

"""Tests for NestedRandomAllocation and BalancedMinSep batch selection."""

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import batch_selection as core_batch_selection
from research.monte_carlo import batch_selection
import numpy as np


def _participation_iters(batches, num_examples):
  """Returns, for each example, the sorted list of iterations it appears in."""
  iters = [[] for _ in range(num_examples)]
  for step, batch in enumerate(batches):
    for idx in batch:
      iters[idx].append(step)
  return iters


def _per_iteration_counts(batches):
  """Returns the batch size of each iteration."""
  return np.array([len(batch) for batch in batches])


class NestedRandomAllocationTest(parameterized.TestCase):

  def test_basic_construction(self):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=4, num_participations=3, iterations=100
    )
    self.assertEqual(strategy.cycle_length, 4)
    self.assertEqual(strategy.num_participations, 3)
    self.assertEqual(strategy.iterations, 100)

  def test_is_batch_selection_strategy(self):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=4, num_participations=3, iterations=100
    )
    self.assertIsInstance(strategy, core_batch_selection.BatchSelectionStrategy)

  @parameterized.parameters(
      (1, 3, 100),
      (4, 3, 100),
      (8, 5, 96),
      (10, 10, 100),  # K == floor(T / b): every candidate is selected.
  )
  def test_exact_participation_count(self, b, k, t):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, t)
    counts = np.zeros(num_examples, dtype=int)
    for batch in batches:
      for idx in batch:
        counts[idx] += 1
    np.testing.assert_array_equal(counts, k)

  @parameterized.parameters((4, 3, 100), (8, 5, 96), (3, 3, 90))
  def test_participations_share_residue_class(self, b, k, t):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=1))
    for iters in _participation_iters(batches, num_examples):
      self.assertLen(iters, k)
      residues = {i % b for i in iters}
      self.assertLen(residues, 1)

  @parameterized.parameters((4, 3, 100), (8, 5, 96), (5, 4, 100))
  def test_min_separation(self, b, k, t):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=2))
    for iters in _participation_iters(batches, num_examples):
      self.assertGreaterEqual(np.diff(iters).min(), b)

  def test_element_range_and_signed_dtype(self):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=4, num_participations=3, iterations=100
    )
    num_examples = 50
    batches = list(strategy.batch_iterator(num_examples, rng=3))
    for batch in batches:
      self.assertTrue(np.issubdtype(batch.dtype, np.signedinteger))
      self.assertGreaterEqual(batch.min(initial=0), 0)
      self.assertLess(batch.max(initial=0), num_examples)

  def test_reproducible_with_same_seed(self):
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=4, num_participations=3, iterations=50
    )
    first = list(strategy.batch_iterator(100, rng=7))
    second = list(strategy.batch_iterator(100, rng=7))
    for a, b in zip(first, second):
      np.testing.assert_array_equal(sorted(a), sorted(b))

  def test_sampling_prob_one_is_noop(self):
    kwargs = dict(cycle_length=4, num_participations=3, iterations=50)
    default = list(
        batch_selection.NestedRandomAllocation(**kwargs).batch_iterator(
            200, rng=11
        )
    )
    explicit = list(
        batch_selection.NestedRandomAllocation(
            sampling_prob=1.0, **kwargs
        ).batch_iterator(200, rng=11)
    )
    for a, b in zip(default, explicit):
      np.testing.assert_array_equal(a, b)

  def test_subsampled_participation_counts(self):
    b, k, t, q = 4, 8, 96, 0.5
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t, sampling_prob=q
    )
    num_examples = 20000
    counts = np.array([
        len(iters)
        for iters in _participation_iters(
            list(strategy.batch_iterator(num_examples, rng=0)), num_examples
        )
    ])
    # Never exceeds K, and averages K * q (Binomial(K, q)).
    self.assertLessEqual(counts.max(), k)
    self.assertAlmostEqual(counts.mean(), k * q, delta=0.05)

  def test_subsampling_preserves_structure(self):
    b, k, t, q = 8, 5, 96, 0.6
    strategy = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t, sampling_prob=q
    )
    num_examples = 5000
    for iters in _participation_iters(
        list(strategy.batch_iterator(num_examples, rng=1)), num_examples
    ):
      if len(iters) > 1:
        # Dropping participations only widens gaps, so min-sep still holds.
        self.assertGreaterEqual(np.diff(iters).min(), b)
      # All surviving participations still share a single residue class.
      self.assertLessEqual(len({i % b for i in iters}), 1)


class BalancedMinSepTest(parameterized.TestCase):

  def test_basic_construction(self):
    strategy = batch_selection.BalancedMinSep(
        min_sep=5, num_participations=3, iterations=100
    )
    self.assertEqual(strategy.min_sep, 5)
    self.assertEqual(strategy.num_participations, 3)
    self.assertEqual(strategy.iterations, 100)
    self.assertFalse(strategy.random_rotation)

  def test_is_batch_selection_strategy(self):
    strategy = batch_selection.BalancedMinSep(
        min_sep=5, num_participations=3, iterations=100
    )
    self.assertIsInstance(strategy, core_batch_selection.BatchSelectionStrategy)

  @parameterized.product(
      random_rotation=[False, True],
      config=[(2, 3, 100), (5, 5, 100), (10, 10, 100), (1, 1, 100)],
  )
  def test_exact_participation_count(self, random_rotation, config):
    b, k, t = config
    strategy = batch_selection.BalancedMinSep(
        min_sep=b,
        num_participations=k,
        iterations=t,
        random_rotation=random_rotation,
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=0))
    self.assertLen(batches, t)
    counts = np.zeros(num_examples, dtype=int)
    for batch in batches:
      for idx in batch:
        counts[idx] += 1
    np.testing.assert_array_equal(counts, k)

  @parameterized.product(
      random_rotation=[False, True],
      config=[(5, 3, 100), (10, 5, 100), (3, 2, 50)],
  )
  def test_linear_min_separation(self, random_rotation, config):
    b, k, t = config
    strategy = batch_selection.BalancedMinSep(
        min_sep=b,
        num_participations=k,
        iterations=t,
        random_rotation=random_rotation,
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=1))
    for iters in _participation_iters(batches, num_examples):
      self.assertLen(iters, k)
      self.assertGreaterEqual(np.diff(iters).min(), b)

  @parameterized.parameters((8, 4, 64), (8, 8, 128), (3, 2, 50))
  def test_random_rotation_cyclic_min_separation(self, b, k, t):
    """With random_rotation=True the wrap-around gap must also be >= b."""
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t, random_rotation=True
    )
    num_examples = 200
    batches = list(strategy.batch_iterator(num_examples, rng=2))
    for iters in _participation_iters(batches, num_examples):
      iters = sorted(iters)
      self.assertLen(iters, k)
      self.assertGreaterEqual(np.diff(iters).min(), b)
      wrap_gap = iters[0] + t - iters[-1]
      self.assertGreaterEqual(wrap_gap, b)

  @parameterized.product(random_rotation=[False, True])
  def test_element_range_and_signed_dtype(self, random_rotation):
    strategy = batch_selection.BalancedMinSep(
        min_sep=5,
        num_participations=3,
        iterations=100,
        random_rotation=random_rotation,
    )
    num_examples = 50
    batches = list(strategy.batch_iterator(num_examples, rng=3))
    for batch in batches:
      self.assertTrue(np.issubdtype(batch.dtype, np.signedinteger))
      self.assertGreaterEqual(batch.min(initial=0), 0)
      self.assertLess(batch.max(initial=0), num_examples)

  def test_random_rotation_flattens_marginal(self):
    """random_rotation=True flattens the per-iteration batch size."""
    b, k, t = 8, 4, 64
    num_examples = 20000
    non_uniform = list(
        batch_selection.BalancedMinSep(
            min_sep=b, num_participations=k, iterations=t, random_rotation=False
        ).batch_iterator(num_examples, rng=0)
    )
    uniform = list(
        batch_selection.BalancedMinSep(
            min_sep=b, num_participations=k, iterations=t, random_rotation=True
        ).batch_iterator(num_examples, rng=0)
    )
    cv_non_uniform = _per_iteration_counts(non_uniform).std() / (
        _per_iteration_counts(non_uniform).mean()
    )
    cv_uniform = _per_iteration_counts(uniform).std() / (
        _per_iteration_counts(uniform).mean()
    )
    # The non-uniform marginal has ~20% edge spikes; the balanced one is flat
    # up to sampling noise, so its coefficient of variation is much smaller.
    self.assertLess(cv_uniform, cv_non_uniform / 2)

  def test_min_sep_one_matches_random_allocation_marginal(self):
    """min_sep=1: each example picks K of T uniformly (random allocation)."""
    k, t = 3, 20
    strategy = batch_selection.BalancedMinSep(
        min_sep=1, num_participations=k, iterations=t
    )
    num_examples = 500
    batches = list(strategy.batch_iterator(num_examples, rng=5))
    for iters in _participation_iters(batches, num_examples):
      self.assertLen(iters, k)
      self.assertLen(set(iters), k)

  def test_reproducible_with_same_seed(self):
    strategy = batch_selection.BalancedMinSep(
        min_sep=5, num_participations=3, iterations=50, random_rotation=True
    )
    first = list(strategy.batch_iterator(100, rng=9))
    second = list(strategy.batch_iterator(100, rng=9))
    for a, b in zip(first, second):
      np.testing.assert_array_equal(sorted(a), sorted(b))

  def test_sampling_prob_one_is_noop(self):
    kwargs = dict(min_sep=5, num_participations=3, iterations=100)
    default = list(
        batch_selection.BalancedMinSep(**kwargs).batch_iterator(200, rng=13)
    )
    explicit = list(
        batch_selection.BalancedMinSep(
            sampling_prob=1.0, **kwargs
        ).batch_iterator(200, rng=13)
    )
    for a, b in zip(default, explicit):
      np.testing.assert_array_equal(a, b)

  def test_subsampled_participation_counts(self):
    b, k, t, q = 5, 5, 100, 0.4
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t, sampling_prob=q
    )
    num_examples = 20000
    counts = np.array([
        len(iters)
        for iters in _participation_iters(
            list(strategy.batch_iterator(num_examples, rng=0)), num_examples
        )
    ])
    # Never exceeds K, and averages K * q (Binomial(K, q)).
    self.assertLessEqual(counts.max(), k)
    self.assertAlmostEqual(counts.mean(), k * q, delta=0.05)

  def test_subsampling_preserves_min_sep(self):
    b, k, t, q = 10, 5, 100, 0.5
    strategy = batch_selection.BalancedMinSep(
        min_sep=b, num_participations=k, iterations=t, sampling_prob=q
    )
    num_examples = 5000
    for iters in _participation_iters(
        list(strategy.batch_iterator(num_examples, rng=2)), num_examples
    ):
      if len(iters) > 1:
        self.assertGreaterEqual(np.diff(iters).min(), b)


class SubsamplingTest(parameterized.TestCase):

  def test_full_allocation_subsampled_matches_cyclic_poisson(self):
    # With num_participations == iterations // cycle_length every candidate is
    # allocated, so per-participation subsampling reduces to independent
    # (cyclic) Poisson sampling of the residue class.
    b, t, q = 4, 40, 0.5
    k = t // b
    num_examples = 30000
    nested = batch_selection.NestedRandomAllocation(
        cycle_length=b, num_participations=k, iterations=t, sampling_prob=q
    )
    cyclic = core_batch_selection.CyclicPoissonSampling(
        sampling_prob=q,
        iterations=t,
        cycle_length=b,
        partition_type=core_batch_selection.PartitionType.INDEPENDENT,
    )

    def count_hist(strategy):
      counts = np.array([
          len(iters)
          for iters in _participation_iters(
              list(strategy.batch_iterator(num_examples, rng=0)), num_examples
          )
      ])
      return np.bincount(counts, minlength=k + 1) / num_examples

    # Both give a Binomial(k, q) participation-count distribution.
    np.testing.assert_allclose(
        count_hist(nested), count_hist(cyclic), atol=0.02
    )

  @parameterized.parameters(-0.1, 1.1)
  def test_invalid_sampling_prob_raises(self, q):
    with self.assertRaises(AssertionError):
      batch_selection.NestedRandomAllocation(
          cycle_length=2, num_participations=2, iterations=10, sampling_prob=q
      )
    with self.assertRaises(AssertionError):
      batch_selection.BalancedMinSep(
          min_sep=2, num_participations=2, iterations=10, sampling_prob=q
      )


NRA = batch_selection.NestedRandomAllocation
BMS = batch_selection.BalancedMinSep
BMS_P = core_batch_selection.BMinSepSampling
S = batch_selection.Strategy


class BuildStrategyTest(parameterized.TestCase):

  # Strategies that accept any expected_participations <= iterations // bands.
  @parameterized.product(
      strategy=(
          S.RA_CYCLIC_POISSON,
          S.NESTED_RANDOM_ALLOCATION,
          S.BALANCED_MIN_SEP,
          S.BALANCED_MIN_SEP_ROTATED,
          S.MIN_SEP_POISSON,
      ),
      setting=(
          dict(iterations=120, bands=6),
          dict(iterations=100, bands=10),
      ),
      expected_participations=(1.5, 3, 7),
  )
  def test_subsampled_strategies(
      self, strategy, setting, expected_participations
  ):
    built = batch_selection.build_strategy(
        strategy,
        iterations=setting['iterations'],
        expected_participations=expected_participations,
        bands=setting['bands'],
    )
    num_examples = 8000
    counts = np.zeros(num_examples)
    for batch in built.batch_iterator(num_examples, rng=0):
      counts[batch] += 1
    self.assertAlmostEqual(counts.mean(), expected_participations, delta=0.1)

  # Pins the exact strategy (type + params) each name dispatches to. T=120, b=6.
  @parameterized.named_parameters(
      ('balls_in_bins', S.BALLS_IN_BINS, 20, NRA(6, 20, 120)),
      ('ra_cyclic_poisson', S.RA_CYCLIC_POISSON, 3, NRA(6, 20, 120, 3 / 20)),
      ('nested_random', S.NESTED_RANDOM_ALLOCATION, 3, NRA(6, 3, 120)),
      ('balanced_min_sep', S.BALANCED_MIN_SEP, 3, BMS(6, 3, 120)),
      ('rotated', S.BALANCED_MIN_SEP_ROTATED, 3, BMS(6, 3, 120, True)),
      ('min_sep_poisson', S.MIN_SEP_POISSON, 3, BMS_P(3 / 105, 120, 6)),
      ('frac_1_5', S.BALANCED_MIN_SEP, 1.5, BMS(6, 2, 120, False, 0.75)),
      ('frac_2_5', S.BALANCED_MIN_SEP, 2.5, BMS(6, 3, 120, False, 2.5 / 3)),
  )
  def test_build_strategy_returns_expected(
      self, strategy, expected_participations, expected
  ):
    built = batch_selection.build_strategy(
        strategy,
        iterations=120,
        expected_participations=expected_participations,
        bands=6,
    )
    self.assertEqual(built, expected)


if __name__ == '__main__':
  absltest.main()
