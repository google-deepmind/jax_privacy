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

"""Demonstrates integration of jax_privacy.BatchSelectionStrategy <> PyGrain.

This file is meant to demonstrate how the BatchSelectionStrategy API can be
used with pygrain for efficient data loading from on-disk datasets. This example
can be forked for your own use cases for your own use cases, potentially with
some customization, but should work out-of-the-box as well.

By default, this data loader is configured to load all of the data into a single
JAX process. In multi-controller JAX setups, the default behavior means all
processes load all data. To distribute the data loading across multiple jax
processes, use the combination of shard_options=grain.ShardByJaxProcess() and
jax.make_array_from_process_local_data().
"""

import concurrent.futures
import copy
import time
from typing import Any

from absl import app
from absl import flags
import grain.python as grain
import jax
from jax_privacy import batch_selection
import numpy as np
import tensorflow_datasets as tfds
import tqdm


class CustomBatchIterator(grain.DatasetIterator):
  """A PyGrain iterator that uses jax_privacy BatchSelectionStrategy.

  This DatasetIterator yields batches of data from the given dataset, along with
  a boolean mask indicating which examples in the batch are padding examples.
  get_state() and set_state are implemented to allow for easy and lightweight
  checkpointing of this batch iterator.
  """

  def __init__(
      self,
      dataset: grain.RandomAccessDataSource,
      strategy: batch_selection.BatchSelectionStrategy,
      rng: np.random.Generator | int,
      *,
      shard_options: grain.ShardOptions = grain.NoSharding(),
      pad_to_multiple_of: int = 1,
      max_workers: int | None = None,
  ):
    """Initializes the CustomBatchIterator.

    Args:
      dataset: The dataset from which to draw samples from.  Each example in the
        dataset should be a PyTree of numpy arrays with common structure/shapes.
      strategy: A BatchSelectionStrategy defining how batches should be formed.
      rng: The random number generator or seed to use to sample minibatches.
      shard_options: If specified, only a subset of the batch will be loaded
        based on shard_index and shard_count. In multi-controller JAX setups,
        use grain.ShardByJaxProcess() to have each process load a disjoint
        subset of the batch.
      pad_to_multiple_of: If provided, pad the batch to a multiple of this
        number.  Larger values reduces the number of compilations needed in
        downstream JAX code.
      max_workers: The number of workers to use for parallel loading.
    """
    super().__init__()
    self._dataset = dataset
    self._strategy = strategy
    self._shard_options = shard_options
    self._pad_to_multiple_of = pad_to_multiple_of
    if self._pad_to_multiple_of % self._shard_options.shard_count != 0:
      raise ValueError(
          f"{pad_to_multiple_of=} must be a multiple of"
          f" {shard_options.shard_count=}"
      )
    self._iteration = 0
    # We do not want the original rng to be modified, so use deepcopy here.
    self._initial_rng = copy.deepcopy(rng)
    self._batch_generator = self._strategy.batch_iterator(
        num_examples=len(self._dataset), rng=copy.deepcopy(rng)
    )
    # Pre-fetch the first element to use as a template for padding.
    self._padding_element = jax.tree.map(np.empty_like, dataset[0])
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers)

  def _get_element(self, idx):
    if idx == -1:
      return self._padding_element
    return self._dataset[idx]

  def __next__(self) -> tuple[Any, np.ndarray]:
    try:
      indices = batch_selection.pad_to_multiple_of(
          next(self._batch_generator), self._pad_to_multiple_of
      )
      shard_size = len(indices) // self._shard_options.shard_count
      start_idx = self._shard_options.shard_index * shard_size
      indices = indices[start_idx : start_idx + shard_size]
      is_padding_example = indices == -1
    except StopIteration as exc:
      raise StopIteration from exc

    batch_elements = list(self._executor.map(self._get_element, indices))

    self._iteration += 1
    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *batch_elements)
    return batch, is_padding_example

  def get_state(self) -> dict[str, Any]:
    return {
        "iteration": self._iteration,
        "initial_rng": self._initial_rng,
    }

  def set_state(self, state: dict[str, Any]):
    self._iteration = state["iteration"]
    self._initial_rng = state["initial_rng"]
    self._batch_generator = self._strategy.batch_iterator(
        num_examples=len(self._dataset), rng=copy.deepcopy(self._initial_rng)
    )
    # Fast-forward the generator
    for _ in range(self._iteration):
      next(self._batch_generator)


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 128, "Batch size for benchmarking.")
flags.DEFINE_integer("num_batches", 100, "Number of batches to benchmark.")


def main(_):
  dataset = grain.MapDataset.source(
      tfds.data_source(
          "mnist",
          split="train",
          builder_kwargs={"file_format": "array_record"},
      )
  )
  strategy = batch_selection.CyclicPoissonSampling(
      iterations=FLAGS.num_batches,
      sampling_prob=FLAGS.batch_size / len(dataset),
  )

  print(f"Starting benchmark with {FLAGS.batch_size=}, {FLAGS.num_batches=}...")

  iterator = CustomBatchIterator(
      dataset, strategy, rng=42, pad_to_multiple_of=32, max_workers=32
  )

  start_time = time.time()
  count = 0
  batch_sizes = set()
  for batch, _ in tqdm.tqdm(iterator):
    count += 1
    batch_size = jax.tree.leaves(batch)[0].shape[0]
    batch_sizes.add(batch_size)
  end_time = time.time()

  total_time = end_time - start_time
  batches_per_sec = count / total_time
  elements_per_sec = (count * FLAGS.batch_size) / total_time

  print("Benchmark results:")
  print(f"  Total time: {total_time:.4f} seconds")
  print(f"  Throughput: {batches_per_sec:.2f} batches/sec")
  print(f"  Throughput: {elements_per_sec:.2f} elements/sec")
  print(f"  Batch Sizes: {sorted(batch_sizes)}")


if __name__ == "__main__":
  app.run(main)
