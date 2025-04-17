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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.auditing import auditing_utils  # pytype: disable=import-error
import numpy as np


_signed_area = auditing_utils._signed_area_py


_rotations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
_inversions = [[0, 2, 1], [1, 0, 2], [2, 1, 0]]


class AuditingUtilsTest(parameterized.TestCase):

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

  def test_get_convex_hull_bad_dtype(self):
    points = np.zeros((2, 2), dtype=np.float64)
    with self.assertRaisesRegex(ValueError, 'Expected points_np to be of type'):
      auditing_utils.get_convex_hull(points)

  @parameterized.parameters(
      ((),),
      ((1,),),
      ((1, 2),),
      ((2, 1),),
      ((2, 3),),
  )
  def test_get_convex_hull_bad_shape(self, shape):
    points = np.zeros(shape, dtype=np.int64)
    with self.assertRaisesRegex(ValueError, 'Expected 2D array for points_np'):
      auditing_utils.get_convex_hull(points)

  def test_get_convex_hull_unsorted(self):
    points = np.array([[1, 1], [0, 0]])
    with self.assertRaisesRegex(ValueError, 'Expected points_np to be sorted'):
      auditing_utils.get_convex_hull(points)

  def test_get_convex_hull_two_points(self):
    points = np.array([[0, 0], [1, 1]])
    hull = auditing_utils.get_convex_hull(points)
    np.testing.assert_equal(hull, points)

  def test_get_convex_hull_linear(self):
    n = 100
    points = np.stack([range(n), range(n)], axis=1)
    hull = auditing_utils.get_convex_hull(points)
    np.testing.assert_equal(hull, points[[0, -1]])

  def test_get_convex_hull_simple_1(self):
    points = np.array([[0, 0], [0, 2], [3, 2], [3, 5], [5, 5]])
    hull = auditing_utils.get_convex_hull(points)
    np.testing.assert_equal(hull, points[[0, 1, 3, 4]])

  def test_get_convex_hull_simple_2(self):
    points = np.array([[0, 0], [0, 2], [2, 2], [2, 3], [3, 3], [3, 5], [5, 5]])
    hull = auditing_utils.get_convex_hull(points)
    # Should not contain [3, 3], which is dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(hull, points[[0, 1, 5, 6]])

  def test_get_convex_hull_simple_3(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 4], [3, 4], [3, 5], [5, 5]])
    hull = auditing_utils.get_convex_hull(points)
    # Should contain [1, 4], which is not dominated by [0, 2] and [3, 5].
    np.testing.assert_equal(hull, points[[0, 1, 3, 5, 6]])

  def test_get_convex_hull_simple_4(self):
    points = np.array([[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [4, 4]])
    hull = auditing_utils.get_convex_hull(points)
    # Should not contain [1, 3], which is a combination of [0, 2] and [2, 4].
    np.testing.assert_equal(hull, points[[0, 1, 5, 6]])

  @parameterized.named_parameters(
      ('increasing', np.sin, np.pi / 2),
      ('decreasing', np.cos, np.pi / 2),
      ('increasing_and_decreasing', np.sin, np.pi),
  )
  def test_get_convex_hull_convex(self, fn, bound):
    xs = np.linspace(0, bound, 100)
    points = np.stack([xs, fn(xs)], axis=1)
    points = (points * 1_000_000).astype(np.int64)
    hull = auditing_utils.get_convex_hull(points)
    np.testing.assert_equal(hull, points)

  @parameterized.named_parameters(
      ('increasing', lambda x: -np.cos(x), np.pi / 2),
      ('decreasing', lambda x: -np.sin(x), np.pi / 2),
      ('decreasing_and_increasing', lambda x: -np.sin(x), np.pi),
  )
  def test_get_convex_hull_concave(self, fn, bound):
    xs = np.linspace(0, bound, 100)
    points = np.stack([xs, fn(xs)], axis=1)
    points = (points * 1_000_000).astype(np.int64)
    hull = auditing_utils.get_convex_hull(points)
    np.testing.assert_equal(hull, points[[0, -1]])

  @parameterized.parameters(range(10))
  def test_get_convex_hull_random(self, seed):
    n = 100
    rng = np.random.default_rng(seed=0xBAD5EED + seed)
    xs = np.linspace(0, np.pi, n)
    ys = np.sin(xs) + rng.normal(scale=0.1, size=n)
    points = np.stack([xs, ys], axis=1)
    points = (points * 1_000_000).astype(np.int64)
    hull = auditing_utils.get_convex_hull(points)

    # Compare to simple cubic time algorithm.
    is_hull = [False] * n
    for i, j in itertools.combinations(range(n), 2):
      if all(_signed_area(points[i], points[j], p) <= 0 for p in points):
        is_hull[i] = is_hull[j] = True
    expected_hull = points[is_hull]

    np.testing.assert_equal(hull, expected_hull)


if __name__ == '__main__':
  absltest.main()
