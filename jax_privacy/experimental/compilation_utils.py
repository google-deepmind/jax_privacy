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

"""Experimental utilities for handling variable batch sizes."""

import functools

import numpy as np


def optimal_physical_batch_sizes(
    batch_sizes: list[int], num_compilations: int
) -> set[int]:
  r"""Find a set of of compiled batch sizes that minimizes wasted compute.

  Given a list of batch sizes $B_1, ..., B_n$ and and a compilation budget,
  $C$, this function finds compiled batch sizes $M_1, ..., M_C$ that minimizes
  the following objective:

  $ L_(M_1, ..., M_C) = sum_{i=1}^n min_{j : M_j \geq B_i} (M_j - B_i) $

  The term M_j - B_i in this objective represents the wasted compute for
  evaluating gradients for a batch of size M_j when the true batch size is B_i.

  The time complexity of this function is $O(C * b^2)$ where $b$ is the number
  of unique batch sizes in the list. It is currently not highly optimized.

  Args:
    batch_sizes: A list of non-negative integers B_1, ..., B_n.
    num_compilations: A non-negative integer representing the number of unique
      batch sizes to return (and compile downstream functions for).

  Returns:
    A set of integers.
  """
  unique = sorted(set(batch_sizes))

  @functools.lru_cache(maxsize=None)
  def solve(C, p):  # pylint: disable=invalid-name
    # Given C compilations remaining and p smallest batch sizes remaining, find
    # optimal list of compiled batch sizes and its cost.
    if C == 1:
      solution = [unique[p]]
      cost = unique[p] * (p + 1)
      return solution, cost

    best_cost = np.inf
    best_solution = None
    for candidate in range(p):
      current_cost = (p - candidate) * unique[p]
      new_solution, new_cost = solve(C - 1, candidate)
      if current_cost + new_cost < best_cost:
        best_cost = current_cost + new_cost
        best_solution = [unique[p]] + new_solution
    return best_solution, best_cost

  return set(solve(num_compilations, len(unique) - 1)[0])
