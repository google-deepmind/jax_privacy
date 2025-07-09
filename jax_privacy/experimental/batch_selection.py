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
    minibatch_size: The desired size of each minibatch. Minibatches of this
      size will typically be passed into a compiled function that computes
      and accumulates clipped gradients.
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
  last_minibatch[:minibatches[-1].shape[0]] = minibatches[-1]
  permutation = microbatching.compute_early_stopping_order(
      minibatch_size, microbatch_size
  )
  minibatches[-1] = last_minibatch[permutation]
  return minibatches


class BatchSelectionStrategy(abc.ABC):
  """Abstract base class for batch selection strategies.

  A batch selection strategy is a function that takes a random number generator
  and returns an iterator of batches of data indices. The strategy can
  either be deterministic or random, it may produce equal-sized batches or
  variable-sized batches. A given strategy may produce either batches of 1D
  numpy arrays of integers, or 2D arrays of integers. In the latter case,
  it is expected that the gradient for each group (i.e., row) will be computed
  and clipped, rather than each example. Note that the batches of indices, which
  specify which examples contribute in which iterations, are generally
  considered sensitive, and should not be inspected directly.

  This API does not prescribe a specific dataset format, but it is expected
  that the format used supports efficient random access to individual examples.
  """

  @abc.abstractmethod
  def batch_iterator(
      self,
      rng: RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields batches of data indices."""


@dataclasses.dataclass(frozen=True)
class CyclicPoissonSampling(BatchSelectionStrategy):
  """Implements Poisson sampling, possibly with cyclic sampling and truncation.

  This generalizes several common sampling strategies:
  - Shuffling the dataset into b batches and doing multiple epochs over the data
  can be achieved with sampling_prob=1, cycle_length=b, shuffle=True. This is
  equivalent to (k, b)-participation as in https://arxiv.org/abs/2211.06530.
  - Standard DP-SGD with Poisson sampling as in
  https://arxiv.org/abs/1607.00133 can be achieved with cycle_length=1 and an
  appropriate sampling_prob.
  - BandMF with b bands and cyclic Poisson sampling as in
  https://arxiv.org/abs/2306.08153 can be achieved with cycle_length=b and an
  appropriate sampling_prob.
  - Truncated Poisson sampling as in https://arxiv.org/abs/2411.04205 can be
  achieved with truncated_batch_size (set to the maximum batch size determined
  by e.g., the hardware or runtime requirements) and otherwise like DP-SGD with
  (cyclic) Poisson sampling.

  Formal guarantees of the batch_iterator:
  - All batches consist of indices in the range [0, num_examples).
  - Each example only appears in batches with index i such that i % cycle_length
    == j for some fixed j per example.
  - Without truncation, every index independently appears in each batch (where
    it is eligible to participate subject to the previous guarantee) with
    probability sampling_prob.
  - With truncation, if > truncated_batch_size examples appear in a batch under
    the previous guarantee, then we select truncated_batch_size of them
    uniformly at random and discard the rest.
  - If even_partition = True, num_examples % cycle_length examples are
    discarded, i.e. never sampled.

  Attributes:
    num_examples: The number of examples in the dataset.
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

  num_examples: int
  sampling_prob: float
  iterations: int
  truncated_batch_size: int | None = None
  cycle_length: int = 1
  shuffle: bool = False
  even_partition: bool = True

  def batch_iterator(self, rng: RngType = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(self.num_examples)

    indices = np.arange(self.num_examples, dtype=dtype)
    if self.shuffle:
      rng.shuffle(indices)
    if self.even_partition:
      group_size = self.num_examples // self.cycle_length
      indices = indices[:group_size * self.cycle_length]

    partition = np.array_split(indices, self.cycle_length)

    for i in range(self.iterations):
      current_group = partition[i % self.cycle_length]
      sample_size = rng.binomial(n=len(current_group), p=self.sampling_prob)
      if self.truncated_batch_size is not None:
        sample_size = min(sample_size, self.truncated_batch_size)
      yield rng.choice(
          current_group, size=sample_size, replace=False
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
  - Each example appears in all batches with index i such that i % cycle_length
    == j, with j chosen uniformly at random independently for each example, and
    in no other batches.

  Attributes:
    num_examples: The number of examples in the dataset.
    iterations: The number of total iterations / batches to generate.
    cycle_length: The number of batches in a cycle, equivalently the separation
      between two consecutive appearances of the same example.
  """

  num_examples: int
  iterations: int
  cycle_length: int

  def batch_iterator(self, rng: RngType = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(self.num_examples)
    indices = np.arange(self.num_examples, dtype=dtype)
    rng.shuffle(indices)

    bin_sizes = rng.multinomial(
        n=self.num_examples,
        pvals=np.ones(self.cycle_length) / self.cycle_length,
    )
    # Pad bin_sizes so that cumsum's output starts with 0.
    batch_cutoffs = np.cumsum(np.append(0, bin_sizes))

    for i in range(self.iterations):
      start_index = batch_cutoffs[i % self.cycle_length]
      end_index = batch_cutoffs[(i % self.cycle_length) + 1]
      yield indices[start_index:end_index]
