# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
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

from collections.abc import Set
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.experimental import compilation_utils
import numpy as np


def brute_force_optimal_batch_sizes(
    batch_sizes: list[int], num_compilations: int
) -> Set[int]:
  """Brute force implementation of optimal_physical_batch_sizes."""
  unique = np.array(sorted(set(batch_sizes)))

  def cost(solution):
    if solution[-1] != unique[-1]:
      return np.inf
    solution = np.array(solution)
    bins = np.digitize(unique, solution, right=True)
    return np.bincount(bins) @ solution

  candidates = itertools.combinations(unique, r=num_compilations)
  solution = min(candidates, key=cost)
  return set(int(x) for x in solution)


class OptimalBatchSizesTest(parameterized.TestCase):

  def test_large_compilations(self):
    batch_sizes = [100, 200, 300, 400, 500]
    optimal_batch_sizes = compilation_utils.optimal_physical_batch_sizes(
        batch_sizes, num_compilations=5
    )
    self.assertLen(optimal_batch_sizes, 5)
    self.assertEqual(optimal_batch_sizes, set(batch_sizes))

  def test_one_compilation(self):
    batch_sizes = [100, 200, 300, 400, 500]
    optimal_batch_sizes = compilation_utils.optimal_physical_batch_sizes(
        batch_sizes, num_compilations=1
    )
    self.assertEqual(optimal_batch_sizes, {500})

  def test_obvious_solution(self):
    batch_sizes = [100, 101, 102, 103, 104, 105, 500]
    optimal_batch_sizes = compilation_utils.optimal_physical_batch_sizes(
        batch_sizes, num_compilations=2
    )
    self.assertEqual(optimal_batch_sizes, {105, 500})

  @parameterized.parameters(range(32))
  def test_matches_brute_force(self, seed):

    prng = np.random.default_rng(seed)
    batch_sizes = prng.choice(16, size=5, replace=False) + 1
    num_compilations = prng.choice(np.arange(5) + 1)

    actual_batch_sizes = compilation_utils.optimal_physical_batch_sizes(
        batch_sizes, num_compilations=num_compilations
    )
    expected_batch_sizes = brute_force_optimal_batch_sizes(
        batch_sizes, num_compilations=num_compilations
    )
    self.assertEqual(actual_batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
  absltest.main()
