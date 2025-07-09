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

import math

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.experimental import batch_selection
import numpy as np


def _get_unique_elements(batches):
  """Returns the set of unique elements from a list of batches."""
  return np.unique_values(np.concatenate(batches))


def _check_batch_sizes_equal(batches, min_batch_size: int, max_batch_size: int):
  """Checks that all batches in the list have the expected size."""
  for batch in batches:
    assert min_batch_size <= len(batch) <= max_batch_size


def _check_no_repeated_indices(batches):
  """Checks that no index appears in multiple batches."""
  combined = np.concatenate(batches)
  assert len(combined) == len(set(combined))


def _check_cyclic_property(batches, cycle_length):
  """Checks that the batches have the cyclic property."""
  for i in range(cycle_length, len(batches)):
    np.testing.assert_array_equal(
        sorted(batches[i]), sorted(batches[i - cycle_length])
    )
  _check_subset_of_kb_participation(batches, cycle_length)


def _check_subset_of_kb_participation(batches, cycle_length):
  """Checks each element can only participate in every cycle_length-th round."""
  partition = []
  for i in range(cycle_length):
    partition.append(_get_unique_elements(batches[i::cycle_length]))
  for i in range(cycle_length):
    for j in range(i + 1, cycle_length):
      assert np.intersect1d(partition[i], partition[j]).size == 0


def _check_element_range(batches, num_examples):
  """Checks that all batches are in the range [0, num_examples)."""
  for batch in batches:
    assert np.all(batch < num_examples)
    assert np.all(batch >= 0)


class BatchSelectionTest(parameterized.TestCase):

  @parameterized.product(
      shuffle=[True, False],
      even_partition=[True, False],
      num_examples=[10],
      cycle_length=[3],
      iterations=[5],
  )
  def test_cyclic_participation(
      self, *, shuffle, even_partition, num_examples, cycle_length, iterations
  ):
    """Tests the use of CyclicPoissonSampling instantiated to do shuffling."""
    strategy = batch_selection.CyclicPoissonSampling(
        num_examples=num_examples,
        sampling_prob=1.0,
        iterations=iterations,
        cycle_length=cycle_length,
        shuffle=shuffle,
        even_partition=even_partition,
    )
    batches = list(strategy.batch_iterator(rng=0))

    self.assertLen(batches, iterations)
    min_batch_size = num_examples // cycle_length
    max_batch_size = min_batch_size if even_partition else min_batch_size + 1
    _check_batch_sizes_equal(batches, min_batch_size, max_batch_size)
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)
    _check_element_range(batches, num_examples)

  @parameterized.product(
      num_examples=[100],
      iterations=[1000],
      cycle_length=[1, 3],
      expected_batch_size=[3],
      shuffle=[True, False],
      even_partition=[True, False],
      truncated_batch_size=[None, 4],
  )
  def test_poisson_sampling(
      self,
      *,
      num_examples,
      iterations,
      cycle_length,
      expected_batch_size,
      shuffle,
      even_partition,
      truncated_batch_size,
  ):
    """Tests for Poisson sampling, potentially cyclic and truncated."""
    sampling_prob = expected_batch_size / (num_examples // cycle_length)
    strategy = batch_selection.CyclicPoissonSampling(
        num_examples=num_examples,
        sampling_prob=sampling_prob,
        iterations=iterations,
        cycle_length=cycle_length,
        shuffle=shuffle,
        even_partition=even_partition,
        truncated_batch_size=truncated_batch_size,
    )
    batches = list(strategy.batch_iterator(rng=0))

    self.assertLen(batches, iterations)
    min_batch_size = 0
    if truncated_batch_size:
      max_batch_size = truncated_batch_size
    elif even_partition:
      max_batch_size = num_examples // cycle_length
    else:
      max_batch_size = math.ceil(num_examples / cycle_length)
    _check_batch_sizes_equal(batches, min_batch_size, max_batch_size)
    for start_index in range(0, iterations, cycle_length):
      _check_no_repeated_indices(
          batches[start_index : start_index + cycle_length]
      )
    _check_element_range(batches, num_examples)
    _check_subset_of_kb_participation(batches, cycle_length)
    # Make sure elements are discarded when using even partition.
    if even_partition:
      distinct_elements = _get_unique_elements(batches)
      self.assertLessEqual(
          distinct_elements.size, (num_examples // cycle_length) * cycle_length
      )
    # Make sure the expected batch size is close to the average batch size.
    if truncated_batch_size is not None:
      self.assertAlmostEqual(
          sum(len(batch) for batch in batches)
          / (iterations * expected_batch_size),
          1.0,
          delta=4 * (iterations * expected_batch_size) ** 0.5,  # ~4 std dev
      )

  def test_poisson_sampling_with_large_cycle_length(self):
    """Test for Poisson sampling with cycle_length > num_examples."""
    num_examples = 10
    cycle_length = 11
    iterations = 1100
    sampling_prob = 0.5
    strategy = batch_selection.CyclicPoissonSampling(
        num_examples=num_examples,
        sampling_prob=sampling_prob,
        iterations=iterations,
        cycle_length=cycle_length,
        even_partition=False,
    )
    batches = list(strategy.batch_iterator(rng=0))

    self.assertLen(batches, iterations)
    min_batch_size = 0
    max_batch_size = 1
    _check_batch_sizes_equal(batches, min_batch_size, max_batch_size)
    for start_index in range(0, iterations, cycle_length):
      _check_no_repeated_indices(
          batches[start_index : start_index + cycle_length]
      )
    _check_element_range(batches, num_examples)
    _check_subset_of_kb_participation(batches, cycle_length)

  @parameterized.product(
      num_examples=[100],
      cycle_length=[10],
      iterations=[10, 20, 21],
  )
  def test_balls_in_bins_sampling(self, num_examples, cycle_length, iterations):
    """Tests for balls-in-bins."""
    strategy = batch_selection.BallsInBinsSampling(
        num_examples=num_examples,
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    self.assertEqual(
        sum(len(batch) for batch in batches[:cycle_length]),
        num_examples,
    )
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)

  def test_balls_in_bins_sampling_with_large_cycle_length(self):
    """Test for balls-in-bins with cycle_length > num_examples."""
    num_examples = 10
    cycle_length = 20
    iterations = 40
    strategy = batch_selection.BallsInBinsSampling(
        num_examples=num_examples,
        iterations=iterations,
        cycle_length=cycle_length,
    )
    batches = list(strategy.batch_iterator(rng=0))
    self.assertLen(batches, iterations)
    _check_element_range(batches, num_examples)
    self.assertEqual(
        sum(len(batch) for batch in batches[:cycle_length]),
        num_examples,
    )
    _check_no_repeated_indices(batches[:cycle_length])
    _check_cyclic_property(batches, cycle_length)


class BatchPartitionTest(parameterized.TestCase):

  @parameterized.parameters(
      (10, 4, None),
      (10, 4, 2),
      (10, 5, None),
      (10, 1, None),
      (17, 6, 6),
      (17, 6, 3),
      (17, 6, 2),
      (17, 6, 1),
      (17, 6, None),
      (17, 17, None),
  )
  def test_split_and_pad(
      self, global_batch_size, minibatch_size, microbatch_size
  ):
    num_examples = 1000
    indices = np.random.randint(0, num_examples, size=global_batch_size)
    minibatches = batch_selection.split_and_pad_global_batch(
        indices, minibatch_size, microbatch_size
    )
    self.assertLen(minibatches, math.ceil(global_batch_size / minibatch_size))

    for minibatch in minibatches[:-1]:
      self.assertLen(minibatch, minibatch_size)
      self.assertTrue(np.all(minibatch != -1))

    self.assertLen(minibatches[-1], minibatch_size)
    self.assertFalse(np.all(minibatches[-1] == -1))

    new_indices = np.concatenate(minibatches)
    np.testing.assert_array_equal(
        np.sort(new_indices[new_indices != -1]), np.sort(indices)
    )


if __name__ == "__main__":
  absltest.main()
