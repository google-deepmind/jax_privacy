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
from jax_privacy._multi_owner import MultiOwnerGraph
import numpy as np


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

  def test_empty(self):
    data = MultiOwnerGraph(
        example_ids=np.array([], dtype=np.int64),
        user_ids=np.array([], dtype=np.int64),
    )
    self.assertEqual(data.num_examples, 0)
    self.assertEqual(data.num_users, 0)
    self.assertEqual(data.num_edges, 0)

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

  def test_from_user_to_examples_all_empty(self):
    """All users empty produces an empty graph."""
    data = MultiOwnerGraph.from_user_to_examples({0: [], 1: []})
    self.assertEqual(data.num_examples, 0)
    self.assertEqual(data.num_users, 0)
    self.assertEqual(data.num_edges, 0)

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


if __name__ == '__main__':
  absltest.main()
