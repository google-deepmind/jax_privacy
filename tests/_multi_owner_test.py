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

import collections
import time

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import batch_selection
import numpy as np


def _check_multi_owner_b_min_sep(batches, data, min_sep):
  """Checks b-min-sep holds for every user in multi-owner attributed data."""
  ex_to_users = collections.defaultdict(set)
  for i in range(data.num_edges):
    ex_to_users[int(data.example_ids[i])].add(int(data.user_ids[i]))
  user_appearances = collections.defaultdict(list)
  for t, batch in enumerate(batches):
    for ex in batch:
      if int(ex) < 0:
        continue
      for u in ex_to_users[int(ex)]:
        user_appearances[u].append(t)
  for u, appearances in user_appearances.items():
    for i in range(len(appearances)):
      for j in range(i + 1, len(appearances)):
        assert appearances[j] - appearances[i] >= min_sep, (
            f'User {u} in batches {appearances[i]} and {appearances[j]}, '
            f'violating min_sep={min_sep}.'
        )


class MultiOwnerGraphTest(parameterized.TestCase):

  def test_from_owners_per_example(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [1, 2], [2]]
    )
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_from_user_to_examples(self):
    data = batch_selection.MultiOwnerGraph.from_user_to_examples(
        {0: [0, 1], 1: [1, 2], 2: [2]}
    )
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_direct_construction(self):
    data = batch_selection.MultiOwnerGraph(
        example_ids=np.array([0, 0, 1, 1, 2]),
        user_ids=np.array([0, 1, 1, 2, 2]),
    )
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_empty(self):
    data = batch_selection.MultiOwnerGraph(
        example_ids=np.array([], dtype=np.int64),
        user_ids=np.array([], dtype=np.int64),
    )
    self.assertEqual(data.num_examples, 0)
    self.assertEqual(data.num_users, 0)
    self.assertEqual(data.num_edges, 0)

  def test_shape_mismatch_raises(self):
    with self.assertRaises(ValueError):
      batch_selection.MultiOwnerGraph(
          example_ids=np.array([0, 1]),
          user_ids=np.array([0]),
      )

  def test_non_1d_raises(self):
    with self.assertRaises(ValueError):
      batch_selection.MultiOwnerGraph(
          example_ids=np.array([[0, 1]]),
          user_ids=np.array([[0, 1]]),
      )

  def test_from_owners_per_example_renumbers_user_ids(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[100, 200], [200, 300]]
    )
    # User IDs should be renumbered to [0, 3).
    self.assertEqual(data.num_users, 3)
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 3))


class GreedyContributionBoundTest(parameterized.TestCase):

  def _verify_contribution_bound(self, attribution, selected, k):
    """Checks that no user exceeds k appearances among selected examples."""
    sel_mask = np.zeros(attribution.num_examples, dtype=bool)
    sel_mask[selected] = True
    edge_sel = sel_mask[attribution.example_ids]
    counts = np.bincount(
        attribution.user_ids[edge_sel], minlength=attribution.num_users
    )
    np.testing.assert_array_less(
        counts, k + 1, err_msg=f'User exceeded contribution bound k={k}'
    )

  def test_independent_set(self):
    """With k=1, the result is an independent set."""
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [1, 2], [0, 2]]
    )
    selected = batch_selection.greedy_contribution_bound(data, 1)
    self._verify_contribution_bound(data, selected, 1)
    self.assertNotEmpty(selected)

  @parameterized.parameters(1, 2, 3)
  def test_contribution_bound_respected(self, k):
    # Pentagon graph: each example shares users with neighbors.
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    )
    selected = batch_selection.greedy_contribution_bound(data, k)
    self._verify_contribution_bound(data, selected, k)

  def test_single_owner_examples(self):
    """Single-owner examples: each user has 3 examples, k=2."""
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0], [0], [0], [1], [1], [1]]
    )
    selected = batch_selection.greedy_contribution_bound(data, 2)
    self._verify_contribution_bound(data, selected, 2)
    # Should select exactly 2 from each user.
    self.assertLen(selected, 4)

  def test_mixed_single_and_multi_owner(self):
    """Mix of single and multi-owner examples."""
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0], [1], [0, 1], [2], [2, 0]]
    )
    selected = batch_selection.greedy_contribution_bound(data, 1)
    self._verify_contribution_bound(data, selected, 1)

  def test_empty_input(self):
    data = batch_selection.MultiOwnerGraph(
        example_ids=np.array([], dtype=np.int64),
        user_ids=np.array([], dtype=np.int64),
    )
    selected = batch_selection.greedy_contribution_bound(data, 1)
    self.assertEmpty(selected)

  def test_k_zero_returns_empty(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example([[0, 1]])
    selected = batch_selection.greedy_contribution_bound(data, 0)
    self.assertEmpty(selected)

  def test_k_larger_than_max_degree_selects_all(self):
    """If k is very large, all examples should be selected."""
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0], [1], [2]]
    )
    selected = batch_selection.greedy_contribution_bound(data, 100)
    self.assertLen(selected, 3)

  def test_scalability_10m_examples(self):
    """Tests that 10M examples with avg 2 users/example runs quickly."""
    num_examples = 10_000_000
    num_users = num_examples
    rng = np.random.default_rng(42)
    degrees = rng.poisson(2, size=num_examples).clip(min=1)
    example_ids = np.repeat(np.arange(num_examples), degrees)
    user_ids = rng.integers(0, num_users, size=len(example_ids))
    data = batch_selection.MultiOwnerGraph(
        example_ids=example_ids, user_ids=user_ids
    )
    start = time.monotonic()
    selected = batch_selection.greedy_contribution_bound(data, 1)
    elapsed = time.monotonic() - start
    # Should complete in under 60 seconds for 10M examples.
    self.assertLess(elapsed, 60, f'Took {elapsed:.1f}s for 10M examples')
    # Verify contribution bound.
    self._verify_contribution_bound(data, selected, 1)


class MultiOwnerMinSepSamplingTest(parameterized.TestCase):

  def test_basic(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0], [1], [2], [3], [4]]
    )
    strategy = batch_selection.MultiOwnerMinSepSampling(
        attribution=data,
        max_batch_size=3,
        iterations=5,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 5)
    for batch in batches:
      self.assertLessEqual(len(batch), 3)
    _check_multi_owner_b_min_sep(batches, data, 2)

  def test_single_owner_only(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[i % 5] for i in range(20)]
    )
    strategy = batch_selection.MultiOwnerMinSepSampling(
        attribution=data,
        max_batch_size=4,
        iterations=5,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 5)
    _check_multi_owner_b_min_sep(batches, data, 2)

  def test_min_sep_one(self):
    """min_sep=1 allows user in consecutive batches."""
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0], [0], [0], [1], [1], [1]]
    )
    strategy = batch_selection.MultiOwnerMinSepSampling(
        attribution=data,
        max_batch_size=2,
        iterations=3,
        min_sep=1,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 3)
    _check_multi_owner_b_min_sep(batches, data, 1)

  @parameterized.parameters(2, 3, 5)
  def test_various_min_sep(self, min_sep):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[i % 10] for i in range(100)]
    )
    strategy = batch_selection.MultiOwnerMinSepSampling(
        attribution=data,
        max_batch_size=5,
        iterations=10,
        min_sep=min_sep,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    self.assertLen(batches, 10)
    _check_multi_owner_b_min_sep(batches, data, min_sep)

  def test_valid_indices(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example(
        [[0, 1], [2], [3], [0, 3]]
    )
    strategy = batch_selection.MultiOwnerMinSepSampling(
        attribution=data,
        max_batch_size=2,
        iterations=4,
        min_sep=2,
    )
    batches = list(strategy.batch_iterator(data.num_examples))
    for batch in batches:
      if batch.size:
        self.assertTrue(np.all(batch < data.num_examples))

  def test_post_init_validation(self):
    data = batch_selection.MultiOwnerGraph.from_owners_per_example([[0]])
    with self.assertRaises(ValueError):
      batch_selection.MultiOwnerMinSepSampling(
          attribution=data,
          max_batch_size=1,
          iterations=1,
          min_sep=0,
      )
    with self.assertRaises(ValueError):
      batch_selection.MultiOwnerMinSepSampling(
          attribution=data,
          max_batch_size=0,
          iterations=1,
          min_sep=1,
      )


if __name__ == '__main__':
  absltest.main()
