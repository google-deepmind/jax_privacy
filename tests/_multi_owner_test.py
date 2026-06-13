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


from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import _multi_owner
import numpy as np

MultiOwnerGraph = _multi_owner.MultiOwnerGraph


class MultiOwnerGraphTest(parameterized.TestCase):

  def test_from_owners_per_example(self):
    data = MultiOwnerGraph.from_owners_per_example([[0, 1], [1, 2], [2]])
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_from_user_to_examples(self):
    data = MultiOwnerGraph.from_user_to_examples({0: [0, 1], 1: [1, 2], 2: [2]})
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_direct_construction(self):
    data = MultiOwnerGraph(
        example_ids=np.array([0, 0, 1, 1, 2]),
        user_ids=np.array([0, 1, 1, 2, 2]),
    )
    self.assertEqual(data.num_examples, 3)
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_edges, 5)

  def test_shape_mismatch_raises(self):
    with self.assertRaises(ValueError):
      MultiOwnerGraph(example_ids=np.array([0, 1]), user_ids=np.array([0]))

  def test_non_1d_raises(self):
    with self.assertRaises(ValueError):
      MultiOwnerGraph(
          example_ids=np.array([[0, 1]]), user_ids=np.array([[0, 1]])
      )

  def test_from_owners_per_example_renumbers_user_ids(self):
    data = MultiOwnerGraph.from_owners_per_example([[100, 200], [200, 300]])
    # User IDs should be renumbered to [0, 3).
    self.assertEqual(data.num_users, 3)
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 3))

  def test_csr_property(self):
    # Example 0 -> users {0, 1}, Example 1 -> users {1, 2}, Example 2 -> {2}.
    data = MultiOwnerGraph(
        example_ids=np.array([0, 0, 1, 1, 2]),
        user_ids=np.array([0, 1, 1, 2, 2]),
    )
    csr = data.csr
    np.testing.assert_array_equal(csr.degree, [2, 2, 1])
    np.testing.assert_array_equal(csr.indptr, [0, 2, 4, 5])
    # users_of should return the correct user slice for each example.
    np.testing.assert_array_equal(sorted(csr.users_of(0)), [0, 1])
    np.testing.assert_array_equal(sorted(csr.users_of(1)), [1, 2])
    np.testing.assert_array_equal(csr.users_of(2), [2])

  def test_from_owners_per_example_unowned_raises(self):
    """An example with no owners is almost certainly a bug."""
    with self.assertRaises(ValueError):
      MultiOwnerGraph.from_owners_per_example([[0, 1], [], [2]])

  def test_from_user_to_examples_empty_user_skipped(self):
    """Users mapped to empty example lists are silently ignored."""
    data = MultiOwnerGraph.from_user_to_examples({0: [0, 1], 1: [], 2: [1, 2]})
    # User 1 has no examples, so only users 0 and 2 contribute.
    self.assertEqual(data.num_users, 2)
    self.assertEqual(data.num_examples, 3)

  def test_from_owners_per_example_noncontiguous_user_ids(self):
    """Large, non-contiguous raw user IDs are renumbered correctly."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[1000, 5000], [5000, 9999], [9999]]
    )
    self.assertEqual(data.num_users, 3)
    self.assertEqual(data.num_examples, 3)
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 3))

  def test_from_user_to_examples_noncontiguous_ids(self):
    """Large, non-contiguous raw IDs are renumbered correctly."""
    data = MultiOwnerGraph.from_user_to_examples(
        {42: [100, 200], 99: [200, 300]}
    )
    self.assertEqual(data.num_users, 2)
    self.assertEqual(data.num_examples, 3)
    self.assertTrue(np.all(data.example_ids >= 0))
    self.assertTrue(np.all(data.example_ids < 3))
    self.assertTrue(np.all(data.user_ids >= 0))
    self.assertTrue(np.all(data.user_ids < 2))

  def test_duplicate_edges_rejected(self):
    """Duplicate (example, user) edges raise ValueError."""
    with self.assertRaises(ValueError):
      MultiOwnerGraph(example_ids=np.array([0, 0]), user_ids=np.array([1, 1]))


class GreedyContributionBoundTest(parameterized.TestCase):

  def _verify_no_user_repeated(self, attribution, selected):
    """Checks that no user appears more than once among selected examples."""
    sel_mask = np.zeros(attribution.num_examples, dtype=bool)
    sel_mask[selected] = True
    edge_sel = sel_mask[attribution.example_ids]
    counts = np.bincount(
        attribution.user_ids[edge_sel], minlength=attribution.num_users
    )
    np.testing.assert_array_less(counts, 2, err_msg="User appeared twice")

  def test_single_owner_examples(self):
    """Single-owner examples: each user has 3 examples, only 1 selected."""
    data = MultiOwnerGraph.from_owners_per_example(
        [[0], [0], [0], [1], [1], [1]]
    )
    selected = _multi_owner.greedy_contribution_bound(data)
    self._verify_no_user_repeated(data, selected)
    # With k=1, exactly 1 per user.
    self.assertLen(selected, 2)

  def test_disjoint_users_selects_all(self):
    """If no user is shared, all examples should be selected."""
    data = MultiOwnerGraph.from_owners_per_example([[0], [1], [2]])
    selected = _multi_owner.greedy_contribution_bound(data)
    self.assertLen(selected, 3)

  def test_max_degree_boundary(self):
    """Verifies that max_degree correctly thresholds example selection."""
    data = MultiOwnerGraph.from_owners_per_example([[0], [1], [2, 3], [4, 5]])

    selected_1 = _multi_owner.greedy_contribution_bound(data, max_degree=1)
    np.testing.assert_array_equal(sorted(selected_1), [0, 1])

    selected_2 = _multi_owner.greedy_contribution_bound(data, max_degree=2)
    np.testing.assert_array_equal(sorted(selected_2), [0, 1, 2, 3])

  @parameterized.parameters(
      # (n_examples, n_users, max_owners, max_degree, seed)
      (50, 20, 2, 1, 42),
      (50, 20, 2, 2, 42),
      (100, 30, 3, 1, 0),
      (100, 30, 3, 2, 0),
      (100, 30, 3, 3, 0),
      (200, 50, 5, 1, 7),
      (200, 50, 5, 3, 7),
      (200, 50, 5, 5, 7),
      (100, 100, 4, 3, 123),
      (500, 100, 2, 1, 99),
      (500, 100, 2, 2, 99),
  )
  def test_random_graph_properties(
      self, n_examples, n_users, max_owners, max_degree, seed
  ):
    """Verifies constraint satisfaction, max_degree, and local optimality."""
    data = _create_random_graph(n_examples, n_users, max_owners, seed)
    selected = _multi_owner.greedy_contribution_bound(
        data, max_degree=max_degree
    )

    # 1. Constraint satisfaction: no user is repeated.
    self._verify_no_user_repeated(data, selected)

    # Determine users that are owned by selected examples.
    sel_mask = np.zeros(data.num_examples, dtype=bool)
    sel_mask[selected] = True
    selected_users = np.unique(data.user_ids[sel_mask[data.example_ids]])

    # 2. Constraint satisfaction: selected examples satisfy max_degree.
    for ex in selected:
      self.assertLessEqual(
          len(data.csr.users_of(ex)),
          max_degree,
          msg=f"Selected example {ex} has degree > {max_degree}",
      )

    # 3. Local optimality: no example can be added to the selection.
    for ex in range(data.num_examples):
      users = data.csr.users_of(ex)
      if len(users) <= max_degree:
        if not np.any(np.isin(users, selected_users)):
          self.assertIn(
              ex,
              selected,
              msg=f"Example {ex} could have been selected but was not.",
          )


def _create_random_graph(n_examples, n_users, max_owners, seed):
  """Generates a random MultiOwnerGraph for testing."""
  rng = np.random.RandomState(seed)
  owners_per_example = []
  for _ in range(n_examples):
    n_owners = rng.randint(1, max_owners + 1)
    owners = rng.choice(n_users, size=n_owners, replace=False)
    owners_per_example.append(owners.tolist())
  return MultiOwnerGraph.from_owners_per_example(owners_per_example)


if __name__ == "__main__":
  absltest.main()
