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

"""PyGrain MapDataset integration for jax_privacy batch selection."""

from collections.abc import Iterator
import concurrent.futures
import copy
from typing import Any
import zlib

import grain.python as grain
import jax
from jax.experimental import multihost_utils
from jax_privacy import batch_selection
import numpy as np


class CustomBatchIterator(grain.DatasetIterator):
  """A PyGrain iterator that uses jax_privacy BatchSelectionStrategy.

  This DatasetIterator yields batches of data from the given dataset, along with
  a boolean mask indicating which examples in the batch are padding examples.
  ``get_state`` and ``set_state`` are implemented to allow for easy and
  lightweight checkpointing of this batch iterator.

  Formal Guarantees:
    - Non-padding examples are sampled according to ``strategy.batch_iterator``,
      starting from ``initial_step``.
    - Padding positions (where ``indices == -1``) are zeroed out in ``batch``
      (having no dependence on ``dataset``) and marked ``True`` in
      ``is_padding_example``.
  """

  def __init__(
      self,
      dataset: grain.RandomAccessDataSource,
      strategy: batch_selection.BatchSelectionStrategy,
      rng: batch_selection.RngType = None,
      *,
      shard_options: grain.ShardOptions = grain.NoSharding(),
      pad_to_multiple_of: int = 1,
      microbatch_size: int | None = None,
      initial_step: int = 0,
      max_workers: int | None = None,
  ):
    """Initializes the CustomBatchIterator.

    Args:
      dataset: The dataset from which to draw samples from. Each example in the
        dataset should be a PyTree of numpy arrays with common structure/shapes.
      strategy: A BatchSelectionStrategy defining how batches should be formed.
      rng: The random number generator or seed to use to sample minibatches.
      shard_options: If specified, only a subset of the batch will be loaded
        based on shard_index and shard_count. In multi-controller JAX setups,
        use grain.ShardByJaxProcess() to have each process load a disjoint
        subset of the batch.
      pad_to_multiple_of: If provided, pad the batch to a multiple of this
        number. Larger values reduces the number of compilations needed in
        downstream JAX code.
      microbatch_size: Optional microbatch size passed to ``pad_to_multiple_of``
        so padding is interleaved across microbatches.
      initial_step: Initial step to fast-forward the batch generator to when
        resuming from a checkpoint.
      max_workers: The number of workers to use for parallel loading. The
        behavior of the default ``max_workers=None`` is version-dependent, and
        typically depends on the number of available CPU cores. See
        `ThreadPoolExecutor
        <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`_
        for more details.
    """
    super().__init__()
    self._dataset = dataset
    self._strategy = strategy
    self._shard_options = shard_options
    self._pad_to_multiple_of = pad_to_multiple_of
    self._microbatch_size = microbatch_size
    if self._pad_to_multiple_of % self._shard_options.shard_count != 0:
      raise ValueError(
          f"{pad_to_multiple_of=} must be a multiple of"
          f" {shard_options.shard_count=}"
      )
    # If rng is used outside of this iterator, our checkpointing logic will
    # fail. We therefore make a deepcopy here to avoid this.
    self.set_state(
        {"iteration": initial_step, "initial_rng": copy.deepcopy(rng)}
    )
    # Pre-fetch the first element to use as a zeroed template for padding.
    self._padding_element = jax.tree.map(np.zeros_like, dataset[0])
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers)
    self._max_workers = max_workers

  def _get_element(self, idx):
    # It might be better in some cases to use batched indexing (via grains
    # private interface SupportsBatchedReadRandomAccessDataSource).
    # In this simple benchmark, we do not see significant performance gains with
    # this, but it may work better in other settings.
    if idx == -1:
      return self._padding_element
    return self._dataset[idx]

  def __next__(self) -> tuple[Any, np.ndarray]:
    try:
      indices = batch_selection.pad_to_multiple_of(
          next(self._batch_generator),
          self._pad_to_multiple_of,
          microbatch_size=self._microbatch_size,
      )
      shard_size = len(indices) // self._shard_options.shard_count
      start_idx = self._shard_options.shard_index * shard_size
      indices = indices[start_idx : start_idx + shard_size]
      is_padding_example = indices == -1
    except StopIteration as exc:
      raise StopIteration from exc

    batch_elements = list(self._executor.map(self._get_element, indices))

    self._iteration += 1
    batch = jax.tree.map(
        lambda *leaves: np.stack(leaves)[1:],
        self._padding_element,
        *batch_elements,
    )
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


def batch_iterator(  # pylint: disable=g-doc-args
    dataset: grain.RandomAccessDataSource,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: batch_selection.RngType = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
) -> Iterator[tuple[Any, jax.Array]]:
  """Yields padded (batch, is_padding_example) pairs from a MapDataset.

  Formal Guarantees:
    - Non-padding examples are sampled according to ``strategy.batch_iterator``,
      starting from ``initial_step``.
    - Padding positions (where ``indices == -1``) are zeroed out in ``batch``
      (having no dependence on ``dataset``) and marked ``True`` in
      ``is_padding_example``.
  """
  if jax.process_count() > 1:
    crcs = [zlib.crc32(x.tobytes()) for x in jax.tree.leaves(dataset[0])]
    seed = copy.deepcopy(np.random.default_rng(rng)).integers(2**63)
    multihost_utils.assert_equal(
        (seed, *crcs), "dataset and rng must match across all JAX processes"
    )

  def _to_device(x: np.ndarray) -> jax.Array:
    if sharding is None:
      return jax.device_put(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])

  iterator = CustomBatchIterator(
      dataset,
      strategy,
      rng=rng,
      pad_to_multiple_of=pad_to_multiple_of,
      microbatch_size=microbatch_size,
      initial_step=initial_step,
  )
  for batch, is_padding in iterator:
    yield jax.tree.map(_to_device, batch), _to_device(is_padding)
