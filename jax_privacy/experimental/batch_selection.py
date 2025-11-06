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

"""API and implementations for batch selection strategies.

The BatchSelectionStrategy API specifies how batches should be formed from
examples in a framework-agnostic manner. It produces indices into a dataset,
not example elements themselves, and hence relies on having a way to access
individual examples by index efficiently, such as with an in-memory list, array,
or pytree of arrays. For datasets which do not fit in memory, we recommend using
pygrain (https://github.com/google/grain), or using an offline job to reorder
the data on disk before loading into your training pipeline.

The implementations in this file generally materialize a vector of indices of
size `num_examples` and hence it requires that this object fits in memory,
i.e., roughly that num_examples < 1e9.
"""

import abc
import dataclasses
import itertools
from typing import Iterator

from jax_privacy.experimental import microbatching
import numpy as np


RngType = np.random.Generator | int | None


def split_and_pad_global_batch(
    indices: np.ndarray, minibatch_size: int, microbatch_size: int | None = None
) -> list[np.ndarray]:
  """Splits a global batch of indices into a list of fixed-size minibatches.

  The last minibatch will be padded with `-1` indices to make it the right size.
  It is crucial that downstream users correctly account for this by e.g.,
  loading in a dummy example not derived from real data, and explicitly passing
  in `is_padding_example` to the clipped gradient function to ensure the
  gradients for these examples are correctly zeroed out.

  Example Usage:
    >>> indices = np.arange(10)
    >>> split_and_pad_global_batch(indices, minibatch_size=4)
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, -1, -1])]

  Args:
    indices: A 1D or 2D numpy array of indices representing the global batch.
    minibatch_size: The desired size of each minibatch. Minibatches of this size
      will typically be passed into a compiled function that computes and
      accumulates clipped gradients.
    microbatch_size: The size of each microbatch. If set, will reorder the last
      minibatch to ensure that the padding indices appear in the right indices
      to enable early stopping within the last minibatch gradient evaluation.
      See `microbatching.compute_early_stopping_order` for more details on this.

  Returns:
    A list of minibatches of indices, each of size exactly minibatch_size.
    The last minibatch may contain extra `-1` indices representing padding
    examples to make it the right size.
  """
  sections = range(minibatch_size, indices.shape[0], minibatch_size)
  minibatches = np.array_split(indices, sections, axis=0)
  minibatch_shape = (minibatch_size,) + indices.shape[1:]
  last_minibatch = np.full(minibatch_shape, -1, dtype=indices.dtype)
  last_minibatch[: minibatches[-1].shape[0]] = minibatches[-1]
  permutation = microbatching.compute_early_stopping_order(
      minibatch_size, microbatch_size
  )
  minibatches[-1] = last_minibatch[permutation]
  return minibatches


def pad_to_multiple_of(indices: np.ndarray, multiple: int) -> np.ndarray:
  """Pads the last dimension of indices to a multiple of multiple.

  Example Usage:
    >>> indices = np.arange(10)
    >>> pad_to_multiple_of(indices, multiple=4)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1])

  Args:
    indices: A 1D array of batch indices.
    multiple: A positive integer. The input batch will be padded to a multiple
      of this value.

  Returns:
    A new 1D array of indices padded with -1.
  """
  if indices.ndim > 1:
    raise ValueError('pad_to_multiple_of currently expects 1D indices.')
  curr_size = indices.shape[0]
  pad_size = (multiple - curr_size) % multiple
  new_indices = np.full(curr_size + pad_size, -1, dtype=indices.dtype)
  new_indices[:curr_size] = indices
  return new_indices


class BatchSelectionStrategy(abc.ABC):
  """Abstract base class for batch selection strategies.

  A batch selection strategy is a function that takes a random number generator
  and returns an iterator of batches of data indices. The strategy can
  either be deterministic or random, it may produce equal-sized batches or
  variable-sized batches. Note that the batches of indices, which
  specify which examples contribute in which iterations, are generally
  considered sensitive, and should not be inspected directly.

  This API does not prescribe a specific dataset format, but it is expected
  that the format used supports efficient random access to individual examples.
  """

  @abc.abstractmethod
  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields 1D batches of data indices."""


@dataclasses.dataclass(frozen=True)
class CyclicPoissonSampling(BatchSelectionStrategy):
  """Implements Poisson sampling, possibly with cyclic sampling and truncation.

  This generalizes several common sampling strategies [1,2,3,4], exemplified
  below (all with expected batch size 3):

  Example Usage (fixed order + multi-epoch) [1]:
    >>> rng = np.random.default_rng(0)
    >>> b = CyclicPoissonSampling(sampling_prob=1, iterations=8, cycle_length=4)
    >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
    [0 1 2] [3 4 5] [6 7 8] [ 9 10 11] [0 1 2] [3 4 5] [6 7 8] [ 9 10 11]

  Example Usage (standard Poisson sampling) [2]:
    >>> b = CyclicPoissonSampling(sampling_prob=0.25, iterations=8)
    >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
    [0 4 9 3 5] [] [5] [4 6 2 7] [ 5 11] [ 2  5  8  6  9 11] [9 1] [7 5 4 3]

  Example Usage (BandMF-style sampling) [3]:
    >>> p = 0.5
    >>> b = CyclicPoissonSampling(sampling_prob=p, iterations=8, cycle_length=2)
    >>> print(*b.batch_iterator(12, rng=rng), sep=' ')
    [2 4 0] [ 9 10 11] [3 5] [9 8] [0 3 4 5] [8] [2 1 4 5] [11]


  References:
  [1] https://arxiv.org/abs/2211.06530
  [2] https://arxiv.org/abs/1607.00133
  [3] https://arxiv.org/abs/2306.08153
  [4] https://arxiv.org/abs/2411.04205

  Formal guarantees of the batch_iterator:
    - All batches consist of indices in the range [0, num_examples).
    - Each example only appears in batches with index i such that i %
      cycle_length == j for some fixed j per example.
    - Without truncation, every index independently appears in each batch
      (where it is eligible to participate subject to the previous
      guarantee) with probability sampling_prob.
    - With truncation, if > truncated_batch_size examples appear in a batch
      under the previous guarantee, then we select truncated_batch_size of
      them uniformly at random and discard the rest.
    - If even_partition = True, num_examples % cycle_length examples are
      discarded, i.e. never sampled.

  Attributes:
    sampling_prob: The probability of sampling an example in rounds when it is
      eligible to participate. To achieve an average batch size of
      expected_batch_size, one should ideally set sampling_prob =
      expected_batch_size / (num_examples // cycle_length).
    iterations: The number of total iterations / batches to generate.
    truncated_batch_size: If True, after Poisson sampling, if we have more than
      truncated_batch_size examples in a batch, we uniformly sample
      truncated_batch_size of them and discard the rest.
    cycle_length: If > 1, we use cyclic Poisson sampling: we partition the
      examples into cycle_length groups, and do Poisson sampling from the groups
      in a round-robin fashion. cycle_length == 1 retrieves standard Poisson
      sampling.
    shuffle: For cyclic Poisson sampling, whether to shuffle the examples before
      discarding (see even_partition) and partitioning.
    even_partition: If True, we discard num_examples % cycle_length examples
      before partitioning in cyclic Poisson sampling. If False, we can have
      uneven partitions. Defaults to True for ease of analysis.
  """

  sampling_prob: float
  iterations: int
  truncated_batch_size: int | None = None
  cycle_length: int = 1
  shuffle: bool = False
  even_partition: bool = True

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(-num_examples)

    indices = np.arange(num_examples, dtype=dtype)
    if self.shuffle:
      rng.shuffle(indices)
    if self.even_partition:
      group_size = num_examples // self.cycle_length
      indices = indices[: group_size * self.cycle_length]

    partition = np.array_split(indices, self.cycle_length)

    for i in range(self.iterations):
      current_group = partition[i % self.cycle_length]
      sample_size = rng.binomial(n=len(current_group), p=self.sampling_prob)
      if self.truncated_batch_size is not None:
        sample_size = min(sample_size, self.truncated_batch_size)
      yield rng.choice(
          current_group, size=sample_size, replace=False, shuffle=False
      )


@dataclasses.dataclass(frozen=True)
class BallsInBinsSampling(BatchSelectionStrategy):
  """Implements balls-in-bins sampling.

  In balls-in-bins, each example is independently assigned a 'bin' from 0 to
  cycle_length-1 uniformly at random, and then appears in all rounds i such that
  i % cycle_length == bin. See https://arxiv.org/abs/2410.06266 and
  https://arxiv.org/abs/2412.16802 for more details.

  Formal guarantees of the batch_iterator:
    - All batches consist of indices in the range [0, num_examples).
    - Each example appears in all batches with index i such that i %
      cycle_length == j, with j chosen uniformly at random independently for
      each example, and in no other batches.

  Attributes:
    iterations: The number of total iterations / batches to generate.
    cycle_length: The number of batches in a cycle, equivalently the separation
      between two consecutive appearances of the same example.
  """

  iterations: int
  cycle_length: int

  def batch_iterator(
      self, num_examples: int, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(-num_examples)
    indices = np.arange(num_examples, dtype=dtype)
    rng.shuffle(indices)

    bin_sizes = rng.multinomial(
        n=num_examples,
        pvals=np.ones(self.cycle_length) / self.cycle_length,
    )
    # Pad bin_sizes so that cumsum's output starts with 0.
    batch_cutoffs = np.cumsum(np.append(0, bin_sizes))

    for i in range(self.iterations):
      start_index = batch_cutoffs[i % self.cycle_length]
      end_index = batch_cutoffs[(i % self.cycle_length) + 1]
      yield indices[start_index:end_index]


@dataclasses.dataclass(frozen=True)
class UserSelectionStrategy:
  """Applies base_strategy at the user level, and selects multiple examples per user.

  Each batch returned by the batch_iterator is a 2D array of integer indices,
  where all entries in the same row are examples owned by the same user. The
  examples in this `user-batch` are chosen in a cyclic fashion (maybe after
  shuffling). For example, if a user owns 3 examples [0, 5, 10], then each
  time this user is selected, the batches will be selected from
  [0, 5, 10, 0, 5, 10, 0, 5, 10, ...]. It is expected that the gradient will be
  evaluated and clipped w.r.t. this `user-batch` before being aggregated across
  users.

  Example Usage:
    >>> base_strategy = CyclicPoissonSampling(sampling_prob=1, iterations=5)
    >>> strategy = UserSelectionStrategy(base_strategy, 2)
    >>> user_ids = np.array([0,0,0,1,1,2])
    >>> iterator = strategy.batch_iterator(user_ids)
    >>> print(next(iterator))
    [[0 1]
     [3 4]
     [5 5]]
    >>> print(next(iterator))
    [[2 0]
     [3 4]
     [5 5]]

  Attributes:
    base_strategy: The base batch selection strategy to apply at the user level.
      Will be used to select batches of users from the set of users.
    examples_per_user_per_batch: The number of examples to select for each user
      in each batch. Determines the number of columns in the returned batches.
    shuffle_per_user: Whether to shuffle the examples for each user before
      selecting examples_per_user_per_batch.
  """

  base_strategy: BatchSelectionStrategy
  examples_per_user_per_batch: int = 1
  shuffle_per_user: bool = False

  def batch_iterator(
      self, user_ids: np.ndarray, rng: RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields 2D batches of data indices.

    Args:
      user_ids: A 1D array that maps each example to a user id, where each
        user_id can be an arbitrary integer.
      rng: A random seed or random number generator.

    Yields:
      2D batches of data indices, where users are sampled according to the
      base_strategy and all entries in the same row are examples owned by the
      same selected user.
    """
    rng = np.random.default_rng(rng)
    # inverse contains cleaned ids in the range [0, ..., nunique-1].
    unique, inverse = np.unique(user_ids, return_inverse=True)
    num_users = unique.size
    num_examples = user_ids.size
    dtype = np.min_scalar_type(-num_examples)

    def create_user_generator(user_id):
      # TODO: b/415360727 - this where is suboptimal, as it is O(n) per user_id.
      owned_examples = np.where(inverse == user_id)[0].astype(dtype)
      if self.shuffle_per_user:
        rng.shuffle(owned_examples)
      return itertools.cycle(list(owned_examples))

    user_generators = [create_user_generator(i) for i in range(num_users)]

    examples_per_user = self.examples_per_user_per_batch or 1

    def get_user_batch(user_id):
      generator = user_generators[user_id]
      sample = itertools.islice(generator, examples_per_user)
      return np.array(list(sample))

    for user_batch in self.base_strategy.batch_iterator(num_users, rng):
      yield np.array([get_user_batch(uid) for uid in user_batch])
