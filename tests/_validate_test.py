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

"""Tests for the centralized validation utilities in ``_validate``."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from jax_privacy import _validate


class NonNegativeTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 0.5, 1e9)
  def test_accepts_non_negative(self, value):
    _validate.non_negative(x=value)  # Should not raise.

  @parameterized.parameters(-1, -0.001, -1e9)
  def test_rejects_negative(self, value):
    with self.assertRaisesRegex(ValueError, r"x=.* >= 0"):
      _validate.non_negative(x=value)

  def test_validates_every_keyword(self):
    _validate.non_negative(a=0, b=1, c=2)  # All fine.
    with self.assertRaisesRegex(ValueError, r"b=-1 >= 0"):
      _validate.non_negative(a=1, b=-1, c=2)

  def test_no_arguments_is_a_noop(self):
    _validate.non_negative()


class PositiveTest(parameterized.TestCase):

  @parameterized.parameters(1, 0.001, 1e9)
  def test_accepts_positive(self, value):
    _validate.positive(x=value)

  @parameterized.parameters(0, -1, -0.5)
  def test_rejects_non_positive(self, value):
    with self.assertRaisesRegex(ValueError, r"x=.* > 0"):
      _validate.positive(x=value)

  def test_zero_is_rejected_but_non_negative_accepts_it(self):
    _validate.non_negative(x=0)
    with self.assertRaises(ValueError):
      _validate.positive(x=0)


class InRangeTest(parameterized.TestCase):

  @parameterized.parameters(0.0, 0.5, 1.0)
  def test_accepts_values_in_range_inclusive(self, value):
    _validate.in_range(0, 1, p=value)

  @parameterized.parameters(-0.1, 1.1, 2.0, -5)
  def test_rejects_values_out_of_range(self, value):
    with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
      _validate.in_range(0, 1, p=value)

  def test_validates_every_keyword(self):
    _validate.in_range(0, 1, a=0.1, b=0.9)
    with self.assertRaisesRegex(ValueError, r"b=.* in \[0, 1\]"):
      _validate.in_range(0, 1, a=0.1, b=1.5)


class EqualTest(parameterized.TestCase):

  def test_accepts_equal_value(self):
    _validate.equal(5, x=5)

  def test_rejects_unequal_value(self):
    with self.assertRaisesRegex(ValueError, r"x=4 does not match expected 5"):
      _validate.equal(5, x=4)

  def test_validates_every_keyword(self):
    _validate.equal(3, a=3, b=3)
    with self.assertRaises(ValueError):
      _validate.equal(3, a=3, b=4)


class BatchTest(parameterized.TestCase):

  def test_returns_shared_batch_size(self):
    tree = {"a": jnp.zeros((8, 3)), "b": jnp.ones((8,))}
    self.assertEqual(_validate.batch(tree), 8)

  def test_accepts_single_array(self):
    self.assertEqual(_validate.batch(jnp.zeros((4, 2))), 4)

  @parameterized.parameters(({},), ([],), (None,))
  def test_rejects_empty_pytree(self, tree):
    with self.assertRaisesRegex(ValueError, r"empty or contains no leaves"):
      _validate.batch(tree)

  def test_rejects_non_array_leaf(self):
    with self.assertRaisesRegex(ValueError, r"to be arrays"):
      _validate.batch({"a": jnp.zeros((4,)), "b": "not-an-array"})

  def test_rejects_scalar_leaf(self):
    with self.assertRaisesRegex(ValueError, r"at least one dimension"):
      _validate.batch({"a": jnp.array(3.0)})

  def test_rejects_inconsistent_first_axis(self):
    with self.assertRaisesRegex(ValueError, r"same size along axis 0"):
      _validate.batch({"a": jnp.zeros((4, 2)), "b": jnp.zeros((5, 2))})


class StrategyTest(parameterized.TestCase):

  @parameterized.parameters(1, 3, 5)
  def test_accepts_one_d_array_within_size(self, size):
    _validate.strategy(np.ones(size), max_size=5)

  def test_accepts_list_input(self):
    _validate.strategy([1.0, 2.0, 3.0], max_size=5)

  @parameterized.parameters(2, 3)
  def test_rejects_non_one_d(self, ndim):
    shape = (2,) * ndim
    with self.assertRaisesRegex(ValueError, r"must be a 1D array"):
      _validate.strategy(np.ones(shape), max_size=10)

  def test_rejects_empty_strategy(self):
    with self.assertRaisesRegex(ValueError, r"size must be in"):
      _validate.strategy(np.array([]), max_size=5)

  def test_rejects_strategy_larger_than_max(self):
    with self.assertRaisesRegex(ValueError, r"size must be in"):
      _validate.strategy(np.ones(6), max_size=5)


class MultiOwnerTest(parameterized.TestCase):

  def test_accepts_valid_edge_lists(self):
    _validate.multi_owner([0, 1, 2], [0, 0, 1])

  def test_rejects_non_one_d(self):
    with self.assertRaisesRegex(ValueError, r"must be 1D arrays"):
      _validate.multi_owner([[0, 1]], [0])

  def test_rejects_length_mismatch(self):
    with self.assertRaisesRegex(ValueError, r"same length"):
      _validate.multi_owner([0, 1, 2], [0, 1])

  def test_rejects_empty_graph(self):
    with self.assertRaisesRegex(ValueError, r"empty graphs"):
      _validate.multi_owner([], [])

  def test_rejects_duplicate_pairs(self):
    with self.assertRaisesRegex(ValueError, r"Duplicate"):
      _validate.multi_owner([0, 1, 0], [0, 1, 0])

  def test_accepts_repeated_ids_in_distinct_pairs(self):
    # The same example or user may appear in multiple *distinct* pairs.
    _validate.multi_owner([0, 0, 1], [0, 1, 1])


if __name__ == "__main__":
  absltest.main()
