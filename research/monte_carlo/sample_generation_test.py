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

"""Tests for pure-JAX participation-pattern sampling."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from research.monte_carlo import sample_generation
import numpy as np


def _batch(fn, key, num_samples, *static_args):
  """vmaps ``fn`` over ``num_samples`` PRNG keys, returning positions as numpy."""
  keys = jax.random.split(key, num_samples)
  in_axes = (0,) + (None,) * len(static_args)
  positions, _ = jax.vmap(fn, in_axes=in_axes)(keys, *static_args)
  return np.asarray(positions)


def _batch_positions_and_mask(fn, key, num_samples, *static_args):
  """Like ``_batch`` but returns both ``positions`` and ``poisson_mask``."""
  keys = jax.random.split(key, num_samples)
  in_axes = (0,) + (None,) * len(static_args)
  positions, mask = jax.vmap(fn, in_axes=in_axes)(keys, *static_args)
  return np.asarray(positions), np.asarray(mask)


class NestedRandomAllocationTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(cycle_length=1, num_participations=3, iterations=100),
      dict(cycle_length=4, num_participations=3, iterations=100),
      dict(cycle_length=8, num_participations=5, iterations=96),
      dict(cycle_length=10, num_participations=1, iterations=100),
  )
  def test_pattern_is_valid(self, cycle_length, num_participations, iterations):
    positions = _batch(
        sample_generation.sample_nested_random_allocation,
        jax.random.PRNGKey(0),
        2000,
        cycle_length,
        num_participations,
        iterations,
    )
    self.assertEqual(positions.shape, (2000, num_participations))
    # Exactly K distinct positions, all in range, and sorted.
    self.assertGreaterEqual(positions.min(), 0)
    self.assertLess(positions.max(), iterations)
    if num_participations > 1:
      self.assertGreater(np.diff(positions, axis=1).min(), 0)
    # All participations of an example share one residue class mod b, so every
    # gap is a multiple of cycle_length (>= cycle_length separation).
    residues = positions % cycle_length
    np.testing.assert_array_equal(
        residues, np.broadcast_to(residues[:, :1], residues.shape)
    )

  def test_bin_is_uniform(self):
    # With K == 1 the single position's bin should be ~uniform over the bins.
    cycle_length, iterations = 5, 100
    positions = _batch(
        sample_generation.sample_nested_random_allocation,
        jax.random.PRNGKey(1),
        20000,
        cycle_length,
        1,
        iterations,
    )
    _, counts = np.unique(positions[:, 0] % cycle_length, return_counts=True)
    self.assertLen(counts, cycle_length)
    self.assertLess(counts.max() / counts.min() - 1.0, 0.15)

  def test_sampling_prob_one_is_noop(self):
    key = jax.random.PRNGKey(5)
    pos_default, mask_default = (
        sample_generation.sample_nested_random_allocation(key, 4, 3, 100)
    )
    pos_explicit, mask_explicit = (
        sample_generation.sample_nested_random_allocation(
            key, 4, 3, 100, sampling_prob=1.0
        )
    )
    np.testing.assert_array_equal(
        np.asarray(pos_default), np.asarray(pos_explicit)
    )
    np.testing.assert_array_equal(
        np.asarray(mask_default), np.asarray(mask_explicit)
    )
    # With no subsampling every participation is kept.
    self.assertTrue(np.asarray(mask_default).all())

  def test_subsampling_keeps_fraction_via_mask(self):
    cycle_length, k, t, q = 4, 5, 100, 0.5
    positions, mask = _batch_positions_and_mask(
        functools.partial(
            sample_generation.sample_nested_random_allocation, sampling_prob=q
        ),
        jax.random.PRNGKey(4),
        20000,
        cycle_length,
        k,
        t,
    )
    self.assertEqual(positions.shape, (20000, k))
    self.assertEqual(mask.shape, (20000, k))
    self.assertEqual(mask.dtype, bool)
    # Subsampling leaves the positions untouched: still K in-range, distinct,
    # sorted participations in a single residue class.
    self.assertGreaterEqual(positions.min(), 0)
    self.assertLess(positions.max(), t)
    self.assertGreater(np.diff(positions, axis=1).min(), 0)
    # Each participation is kept independently with probability q.
    self.assertAlmostEqual(mask.mean(), q, delta=0.02)


class BalancedMinSepTest(parameterized.TestCase):

  @parameterized.product(
      random_rotation=[False, True],
      config=[(3, 2, 100), (5, 5, 100), (10, 10, 100), (1, 1, 100)],
  )
  def test_pattern_is_valid(self, random_rotation, config):
    min_sep, num_participations, iterations = config
    positions = _batch(
        sample_generation.sample_balanced_min_sep,
        jax.random.PRNGKey(0),
        2000,
        min_sep,
        num_participations,
        iterations,
        random_rotation,
    )
    self.assertEqual(positions.shape, (2000, num_participations))
    self.assertGreaterEqual(positions.min(), 0)
    self.assertLess(positions.max(), iterations)
    # Exactly K distinct, sorted positions with interior gaps >= min_sep.
    if num_participations > 1:
      self.assertGreaterEqual(np.diff(positions, axis=1).min(), min_sep)
    if random_rotation:
      # The cyclic wrap-around gap must also respect the minimum separation.
      wrap_gap = iterations - positions[:, -1] + positions[:, 0]
      self.assertGreaterEqual(wrap_gap.min(), min_sep)

  def test_random_rotation_marginal_is_flat(self):
    # random_rotation=True makes every iteration equally likely (K / T).
    # (min_sep, K, T) = (2, 2, 4) gives a very non-flat marginal without the
    # rotation ([2/3, 1/3, 1/3, 2/3]), so this fails for most seeds if broken.
    min_sep, num_participations, iterations = 2, 2, 4
    positions = _batch(
        sample_generation.sample_balanced_min_sep,
        jax.random.PRNGKey(2),
        40000,
        min_sep,
        num_participations,
        iterations,
        True,
    )
    counts = np.bincount(positions.reshape(-1), minlength=iterations)
    expected = positions.shape[0] * num_participations / iterations
    self.assertLess(np.abs(counts / expected - 1.0).max(), 0.1)

  def test_no_rotation_marginal_favors_endpoints(self):
    # Without the rotation, (min_sep, K, T) = (2, 2, 4) has marginal
    # [2/3, 1/3, 1/3, 2/3], so the endpoints are ~2x the interior.
    min_sep, num_participations, iterations = 2, 2, 4
    positions = _batch(
        sample_generation.sample_balanced_min_sep,
        jax.random.PRNGKey(3),
        40000,
        min_sep,
        num_participations,
        iterations,
        False,
    )
    counts = np.bincount(positions.reshape(-1), minlength=iterations)
    self.assertGreater(counts[0], 1.5 * counts[1])
    self.assertGreater(counts[-1], 1.5 * counts[-2])

  def test_sampling_prob_one_is_noop(self):
    key = jax.random.PRNGKey(8)
    pos_default, mask_default = sample_generation.sample_balanced_min_sep(
        key, 5, 3, 100, True
    )
    pos_explicit, mask_explicit = sample_generation.sample_balanced_min_sep(
        key, 5, 3, 100, True, sampling_prob=1.0
    )
    np.testing.assert_array_equal(
        np.asarray(pos_default), np.asarray(pos_explicit)
    )
    np.testing.assert_array_equal(
        np.asarray(mask_default), np.asarray(mask_explicit)
    )
    self.assertTrue(np.asarray(mask_default).all())

  def test_subsampling_keeps_fraction_via_mask(self):
    min_sep, k, t, q = 5, 5, 100, 0.6
    positions, mask = _batch_positions_and_mask(
        functools.partial(
            sample_generation.sample_balanced_min_sep, sampling_prob=q
        ),
        jax.random.PRNGKey(6),
        20000,
        min_sep,
        k,
        t,
        False,
    )
    self.assertEqual(positions.shape, (20000, k))
    self.assertEqual(mask.shape, (20000, k))
    # Positions keep their min-sep structure regardless of subsampling.
    self.assertGreaterEqual(np.diff(positions, axis=1).min(), min_sep)
    self.assertAlmostEqual(mask.mean(), q, delta=0.02)


class ApiTest(parameterized.TestCase):

  def test_jittable(self):
    fn = jax.jit(
        functools.partial(
            sample_generation.sample_balanced_min_sep,
            min_sep=3,
            num_participations=4,
            iterations=100,
            random_rotation=True,
        )
    )
    positions, mask = fn(jax.random.PRNGKey(0))
    self.assertEqual(positions.shape, (4,))
    self.assertEqual(mask.shape, (4,))

  def test_deterministic_given_key(self):
    key = jax.random.PRNGKey(7)
    first_pos, first_mask = sample_generation.sample_nested_random_allocation(
        key, 4, 3, 100
    )
    second_pos, second_mask = sample_generation.sample_nested_random_allocation(
        key, 4, 3, 100
    )
    np.testing.assert_array_equal(np.asarray(first_pos), np.asarray(second_pos))
    np.testing.assert_array_equal(
        np.asarray(first_mask), np.asarray(second_mask)
    )


def _dense_banded_c(c_col, n):
  """Dense n x n lower-triangular banded Toeplitz with band c_col."""
  dense_c = np.zeros((n, n))
  for d in range(len(c_col)):
    i = np.arange(n - d)
    dense_c[i + d, i] = c_col[d]
  return dense_c


def _random_min_sep_pattern(rng, k, t, b):
  """K sorted positions in [0, t) with gaps >= b (bands don't overlap)."""
  # Choose K distinct positions in a compressed range, then spread them apart so
  # every gap grows by b - 1, making all gaps >= b.
  compressed = np.sort(rng.choice(t - (k - 1) * (b - 1), size=k, replace=False))
  return compressed + np.arange(k) * (b - 1)


def _all_kept(part_pattern):
  """An all-True poisson_mask aligned with ``part_pattern`` (no subsampling)."""
  return jnp.ones(part_pattern.shape[0], dtype=bool)


class BandedCTimesPatternTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(b=1, k=3, t=20),
      dict(b=4, k=3, t=20),
      dict(b=5, k=5, t=50),
      dict(b=8, k=1, t=30),
      dict(b=6, k=3, t=30),
  )
  def test_matches_dense_reference(self, b, k, t):
    rng = np.random.default_rng(b * 100 + k * 10 + t)
    c_col = rng.standard_normal(b)
    part_pattern = jnp.asarray(_random_min_sep_pattern(rng, k, t, b))
    out = sample_generation.banded_c_times_pattern(
        jnp.asarray(c_col), part_pattern, t, _all_kept(part_pattern)
    )
    x = np.zeros(t)
    x[np.asarray(part_pattern)] = 1.0
    np.testing.assert_allclose(
        np.asarray(out), _dense_banded_c(c_col, t) @ x, rtol=1e-6, atol=1e-6
    )

  def test_single_participation_places_band(self):
    c_col = jnp.array([1.0, 2.0, 3.0])
    pattern = jnp.array([4])
    out = sample_generation.banded_c_times_pattern(
        c_col, pattern, 10, _all_kept(pattern)
    )
    expected = np.zeros(10)
    expected[4:7] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(np.asarray(out), expected)

  def test_contributions_past_end_are_dropped(self):
    c_col = jnp.array([1.0, 2.0, 3.0])
    pattern = jnp.array([9])
    out = sample_generation.banded_c_times_pattern(
        c_col, pattern, 10, _all_kept(pattern)
    )
    expected = np.zeros(10)
    expected[9] = 1.0  # Rows 10 and 11 lie outside C.
    np.testing.assert_allclose(np.asarray(out), expected)

  def test_min_sep_participations_are_independent(self):
    # Two participations exactly b apart: their bands abut but never overlap.
    c_col = jnp.array([1.0, 2.0, 3.0])
    pattern = jnp.array([1, 4])
    out = sample_generation.banded_c_times_pattern(
        c_col, pattern, 10, _all_kept(pattern)
    )
    expected = np.zeros(10)
    expected[1:4] = [1.0, 2.0, 3.0]
    expected[4:7] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(np.asarray(out), expected)

  def test_masked_positions_are_dropped(self):
    # A participation dropped by Poisson subsampling has a False poisson_mask
    # entry and must contribute nothing, matching the dense product taken over
    # only the surviving positions.
    c_col = jnp.array([1.0, 2.0, 3.0])
    t = 20
    part_pattern = jnp.array([2, 8, 14, 17])  # All valid, gaps >= len(c_col).
    poisson_mask = jnp.array([True, False, True, False])
    out = sample_generation.banded_c_times_pattern(
        c_col, part_pattern, t, poisson_mask
    )
    x = np.zeros(t)
    x[[2, 14]] = 1.0  # Only the kept participations contribute.
    np.testing.assert_allclose(
        np.asarray(out), _dense_banded_c(np.asarray(c_col), t) @ x
    )

  def test_jittable_and_vmap_over_patterns(self):
    c_col = jnp.array([1.0, 0.5, 0.25])
    patterns = jnp.array([[0, 3, 6], [1, 4, 7]])
    masks = jnp.array([[True, True, True], [True, False, True]])
    fn = jax.jit(
        lambda c, p, m: sample_generation.banded_c_times_pattern(c, p, 10, m)
    )
    batched = jax.vmap(fn, in_axes=(None, 0, 0))(c_col, patterns, masks)
    self.assertEqual(batched.shape, (2, 10))
    for i in range(patterns.shape[0]):
      single = sample_generation.banded_c_times_pattern(
          c_col, patterns[i], 10, masks[i]
      )
      np.testing.assert_allclose(np.asarray(batched[i]), np.asarray(single))


if __name__ == '__main__':
  absltest.main()
