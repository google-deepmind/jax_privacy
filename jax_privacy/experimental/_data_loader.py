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

"""PyGrain data loader for differentially private training.

This module provides batch iterators that integrate a PyGrain
``MapDataset`` with :mod:`jax_privacy.batch_selection` strategies.

Two loading modes are supported:

* **Preload** (``preload=True`` or auto-detected): Materializes the full
  dataset into memory as a stacked NumPy PyTree, then uses fast numpy
  fancy-indexing per batch.  Best for datasets that comfortably fit in
  host RAM.  In multi-controller JAX setups, use the combination of
  shard_options=grain.ShardByJaxProcess() and
  jax.make_array_from_process_local_data().

* **Streaming** (``preload=False``): Loads elements on-demand using a
  ``ThreadPoolExecutor`` for concurrent reads.  This is the right choice
  for large or remote datasets (e.g., ArrayRecord on disk/remote storage)
  that should not be held entirely in memory.

When ``preload`` is left as ``None`` (the default), the loader estimates
the in-memory size from the first element's PyTree and the dataset length.
If the estimate is under ``_PRELOAD_THRESHOLD_BYTES`` (1 GiB), the dataset
is preloaded; otherwise it streams.

This module is intentionally *not* re-exported from any ``__init__.py`` —
users who do not have PyGrain installed will never import it.
"""

from __future__ import annotations

from collections.abc import Generator
import concurrent.futures
import copy
import logging
from typing import Any

import jax
from jax_privacy import batch_selection
import numpy as np

_PRELOAD_THRESHOLD_BYTES = 1 << 30  # 1 GiB


def is_pygrain_map_dataset(dataset: Any) -> bool:
  """Checks whether ``dataset`` is a PyGrain MapDataset, by class name."""
  for cls in type(dataset).__mro__:
    if cls.__name__ == "MapDataset":
      return True
  return False


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


def _estimate_dataset_bytes(dataset) -> int:
  """Estimates the total in-memory size of the dataset in bytes.

  Computes the size of one element's PyTree leaves (via ``nbytes``) and
  multiplies by the number of elements.

  Args:
    dataset: A PyGrain ``MapDataset`` supporting ``len`` and ``__getitem__``.

  Returns:
    Estimated size in bytes.
  """
  first_element = dataset[0]
  element_bytes = sum(
      jax.tree.leaves(jax.tree.map(lambda x: x.nbytes, first_element))
  )
  return element_bytes * len(dataset)


def _should_preload(dataset) -> bool:
  """Decides whether to preload based on estimated dataset size."""
  estimated = _estimate_dataset_bytes(dataset)
  decision = estimated <= _PRELOAD_THRESHOLD_BYTES
  logging.info(
      "Dataset size estimate: %.2f MiB (%d elements). Preload: %s.",
      estimated / (1 << 20),
      len(dataset),
      decision,
  )
  return decision


# ---------------------------------------------------------------------------
# Preload helpers
# ---------------------------------------------------------------------------


def _preload(dataset, max_workers: int | None = None):
  """Materializes all elements into a stacked PyTree for fast indexing."""
  n = len(dataset)
  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    elements = list(executor.map(dataset.__getitem__, range(n)))
  return jax.tree.map(lambda *leaves: np.stack(leaves), *elements)


def _get_batch_preloaded(stacked, indices):
  """Indexes into a stacked PyTree with padding support."""
  is_padding_example = indices == -1
  # Replace -1 with 0 for safe indexing, then zero out padding entries.
  safe = np.where(is_padding_example, 0, indices)

  def _index_and_zero(x):
    mask = np.expand_dims(is_padding_example, tuple(range(1, x.ndim)))
    return np.where(mask, 0, x[safe])

  return jax.tree.map(_index_and_zero, stacked), is_padding_example


# ---------------------------------------------------------------------------
# Streaming iterator (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


class PrivateBatchIterator:
  """A batch iterator that uses jax_privacy BatchSelectionStrategy.

  This iterator yields batches of data from the given dataset, along with
  a boolean mask indicating which examples in the batch are padding examples.
  ``get_state`` and ``set_state`` are implemented to allow for easy and
  lightweight checkpointing of this batch iterator.
  """

  def __init__(
      self,
      dataset: Any,
      strategy: batch_selection.BatchSelectionStrategy,
      rng: np.random.Generator | int,
      *,
      shard_options: Any = None,
      pad_to_multiple_of: int = 1,
      max_workers: int | None = None,
  ):
    """Initializes the PrivateBatchIterator.

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
      max_workers: The number of workers to use for parallel loading. The
        behavior the default `max_workers=None` is version-dependent, and
        typically depends on the number of available CPU cores. See
        https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
          for more details.
    """
    self._dataset = dataset
    self._strategy = strategy
    self._shard_options = shard_options
    self._pad_to_multiple_of = pad_to_multiple_of
    if (
        self._shard_options is not None
        and self._pad_to_multiple_of % self._shard_options.shard_count != 0
    ):
      raise ValueError(
          f"{pad_to_multiple_of=} must be a multiple of"
          f" {shard_options.shard_count=}"
      )
    self._iteration = 0
    # If rng is used outside of this iterator, our checkpointing logic will
    # fail. We therefore make a deepcopy here to avoid this.
    self._initial_rng = copy.deepcopy(rng)
    self._batch_generator = self._strategy.batch_iterator(
        num_examples=len(self._dataset), rng=copy.deepcopy(rng)
    )
    # Pre-fetch the first element to use as a template for padding.
    self._padding_element = jax.tree.map(np.empty_like, dataset[0])
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

  def __iter__(self):
    return self

  def __next__(self) -> tuple[Any, np.ndarray]:
    indices = batch_selection.pad_to_multiple_of(
        next(self._batch_generator), self._pad_to_multiple_of
    )
    if self._shard_options is not None:
      shard_size = len(indices) // self._shard_options.shard_count
      start_idx = self._shard_options.shard_index * shard_size
      indices = indices[start_idx : start_idx + shard_size]
    if indices.size == 0:
      return self.__next__()
    is_padding_example = indices == -1

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iterate_batches(
    dataset: Any,
    strategy: batch_selection.BatchSelectionStrategy,
    rng: np.random.Generator,
    *,
    shard_options: Any = None,
    pad_to_multiple_of: int = 1,
    preload: bool | None = None,
    max_workers: int | None = None,
) -> Generator[tuple[Any, np.ndarray], None, None]:
  """Yields ``(batch, is_padding_example)`` tuples from a PyGrain MapDataset.

  Args:
    dataset: A PyGrain ``MapDataset`` supporting ``len`` and ``__getitem__``.
    strategy: A ``BatchSelectionStrategy`` that produces index arrays.
    rng: A NumPy random generator for the batch strategy.
    shard_options: If specified, only a subset of the batch will be loaded based
      on shard_index and shard_count. In multi-controller JAX setups, use
      grain.ShardByJaxProcess() to have each process load a disjoint subset of
      the batch.
    pad_to_multiple_of: If provided, pad the batch to a multiple of this number.
      Larger values reduces the number of compilations needed in downstream JAX
      code.
    preload: Whether to materialize the full dataset into memory.  ``True``
      forces preloading, ``False`` forces streaming, and ``None`` (default)
      auto-decides based on estimated dataset size (preloads if < 1 GiB).
    max_workers: Maximum thread pool workers.  Used for parallel element loading
      in *both* preload and streaming modes.

  Yields:
    ``(batch, is_padding_example)`` where ``batch`` is a stacked PyTree and
    ``is_padding_example`` is a boolean array flagging padding entries.
  """
  if preload is None:
    preload = _should_preload(dataset)

  if preload:
    stacked = _preload(dataset, max_workers=max_workers)
    for indices in strategy.batch_iterator(
        len(dataset), rng=copy.deepcopy(rng)
    ):
      indices = batch_selection.pad_to_multiple_of(indices, pad_to_multiple_of)
      if indices.size == 0:
        continue
      yield _get_batch_preloaded(stacked, indices)
  else:
    yield from PrivateBatchIterator(
        dataset,
        strategy,
        rng,
        shard_options=shard_options,
        pad_to_multiple_of=pad_to_multiple_of,
        max_workers=max_workers,
    )
