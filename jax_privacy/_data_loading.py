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

"""Data loading and batch iteration for jax_privacy training loops."""

from collections.abc import Iterator
import functools
import itertools

import jax
from jax_privacy import _validate
from jax_privacy import batch_selection
import numpy as np
import optax


def _from_pytree(
    dataset: optax.ArrayTree,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: batch_selection.RngType = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
) -> Iterator[tuple[optax.ArrayTree, jax.Array]]:
  """Yields padded (batch, is_padding_example) pairs from an array PyTree."""
  num_examples = _validate.batch(dataset)
  _validate.multihost(jax.tree.map(lambda x: x[0], dataset), rng)
  index_iterator = strategy.batch_iterator(num_examples, rng=rng)

  # When sharding across a mesh (including multi-controller JAX),
  # make_array_from_callback slices and transfers only the process-addressable
  # shards to HBM. In multi-controller setups, every process must hold the
  # full dataset and use the same rng so each process samples the same global
  # batch indices before slicing its local device shards.
  def _to_device(x: np.ndarray) -> jax.Array:
    if sharding is None:
      return jax.device_put(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])

  def _index_and_zero(x, indices):
    mask = np.expand_dims(indices == -1, tuple(range(1, x.ndim)))
    return _to_device(np.where(mask, 0, x[indices]))

  # Fast-forward the index iterator to initial_step when resuming from a
  # checkpoint, without materializing or transferring skipped batches.
  # TODO: Investigate stateful iterator instead of rng replay.
  for indices in itertools.islice(index_iterator, initial_step, None):
    indices = batch_selection.pad_to_multiple_of(
        indices, pad_to_multiple_of, microbatch_size=microbatch_size
    )
    fn = functools.partial(_index_and_zero, indices=indices)
    yield jax.tree.map(fn, dataset), _to_device(indices == -1)


def batch_iterator(
    dataset: optax.ArrayTree,
    strategy: batch_selection.BatchSelectionStrategy,
    *,
    rng: batch_selection.RngType = None,
    pad_to_multiple_of: int = 1,
    microbatch_size: int | None = None,
    initial_step: int = 0,
    sharding: jax.sharding.NamedSharding | None = None,
) -> Iterator[tuple[optax.ArrayTree, jax.Array]]:
  """Returns an iterator of padded (batch, is_padding_example) pairs.

  Formal Guarantees:
    - Non-padding examples are sampled according to ``strategy.batch_iterator``,
      starting from ``initial_step``.
    - Padding positions (where ``indices == -1``) are zeroed out in ``batch``
      (having no dependence on ``dataset``) and marked ``True`` in
      ``is_padding_example``.

  Args:
    dataset: A PyTree of arrays where the first axis of each leaf is the batch /
      example dimension.
    strategy: Batch selection strategy producing per-step example indices.
    rng: Optional random seed or ``numpy.random.Generator`` for batch sampling.
    pad_to_multiple_of: Pads each batch size to a multiple of this integer.
    microbatch_size: Optional microbatch size used to reorder padding examples
      for early stopping.
    initial_step: Step index to fast-forward the batch iterator to when resuming
      from a checkpoint.
    sharding: Optional ``NamedSharding`` for placing batches across a device
      mesh.

  Returns:
    An iterator yielding ``(batch, is_padding_example)`` pairs on device.
  """
  kwargs = dict(
      rng=rng,
      pad_to_multiple_of=pad_to_multiple_of,
      microbatch_size=microbatch_size,
      initial_step=initial_step,
      sharding=sharding,
  )
  return _from_pytree(dataset, strategy, **kwargs)
