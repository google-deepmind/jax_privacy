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

"""Multi-owner differential privacy support for batch selection.

Provides data structures and algorithms for the multi-owner model of
differential privacy, where examples may be attributed to multiple users.

References: https://arxiv.org/abs/2503.03622
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import dataclasses

from jax_privacy import batch_selection
import numpy as np
from numpy import typing as npt


@dataclasses.dataclass(frozen=True)
class MultiOwnerGraph:
  """Bipartite edge list of (example, user) attribution pairs.

  Entry ``(example_ids[i], user_ids[i])`` means that example
  ``example_ids[i]`` is attributed to user ``user_ids[i]``.

  For smaller datasets, use ``from_owners_per_example`` or
  ``from_user_to_examples``.

  Example Usage:
    >>> data = MultiOwnerGraph.from_owners_per_example([[0, 1], [1, 2], [2]])
    >>> data.num_examples
    3
    >>> data.num_users
    3
    >>> data.num_edges
    5

  References: https://arxiv.org/abs/2503.03622

  Attributes:
    example_ids: 1D integer array of example indices in ``[0, num_examples)``.
    user_ids: 1D integer array of user indices in ``[0, num_users)``, aligned
      with ``example_ids``.
  """

  example_ids: np.ndarray
  user_ids: np.ndarray

  def __post_init__(self):
    if self.example_ids.ndim != 1 or self.user_ids.ndim != 1:
      raise ValueError('example_ids and user_ids must be 1D arrays.')
    if len(self.example_ids) != len(self.user_ids):
      raise ValueError(
          'example_ids and user_ids must have the same length, got '
          f'{len(self.example_ids)} and {len(self.user_ids)}.'
      )

  @property
  def num_examples(self) -> int:
    """Number of distinct examples (``max(example_ids) + 1``)."""
    return int(self.example_ids.max()) + 1 if self.num_edges else 0

  @property
  def num_users(self) -> int:
    """Number of distinct users (``max(user_ids) + 1``)."""
    return int(self.user_ids.max()) + 1 if self.num_edges else 0

  @property
  def num_edges(self) -> int:
    """Total number of (example, user) attribution edges."""
    return len(self.example_ids)

  @classmethod
  def from_owners_per_example(
      cls, owners_per_example: Sequence[npt.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a sequence of user sets, one per example.

    Example IDs are assigned sequentially (0, 1, ...). User IDs are
    renumbered to ``[0, num_users)``.

    Args:
      owners_per_example: ``owners_per_example[i]`` is the array-like of user
        IDs attributed to example ``i``.

    Returns:
      A new MultiOwnerGraph instance.
    """
    arrays = [np.asarray(o, dtype=np.int64) for o in owners_per_example]
    sizes = np.array([len(a) for a in arrays])
    if int(sizes.sum()) == 0:
      return cls(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    example_ids = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    user_ids = np.concatenate(arrays)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  @classmethod
  def from_user_to_examples(
      cls, user_to_examples: Mapping[int, npt.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a mapping of user IDs to their example sets.

    Both example and user IDs are renumbered to contiguous ranges.

    Args:
      user_to_examples: ``user_to_examples[u]`` is the array-like of example IDs
        attributed to user ``u``.

    Returns:
      A new MultiOwnerGraph instance with renumbered IDs.
    """
    ex_arrays, u_arrays = [], []
    for uid, examples in user_to_examples.items():
      ex_arr = np.asarray(examples, dtype=np.int64)
      ex_arrays.append(ex_arr)
      u_arrays.append(np.full(len(ex_arr), uid, dtype=np.int64))
    if not ex_arrays:
      return cls(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    example_ids = np.concatenate(ex_arrays)
    user_ids = np.concatenate(u_arrays)
    _, example_ids = np.unique(example_ids, return_inverse=True)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  def _build_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(degree, indptr, sorted_users)`` for per-example user lookup."""
    n = self.num_examples
    degree = np.bincount(self.example_ids, minlength=n)
    order = np.argsort(self.example_ids, kind='stable')
    users = self.user_ids[order]
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    return degree, indptr, users


def _users_of(indptr: np.ndarray, sorted_users: np.ndarray, ex: int):
  """Returns the slice of users attributed to example ``ex``."""
  return sorted_users[indptr[ex] : indptr[ex + 1]]


def greedy_contribution_bound(
    attribution: MultiOwnerGraph,
    contribution_bound: int,
) -> np.ndarray:
  """Selects examples so each user appears at most ``contribution_bound`` times.

  Processes examples in ascending degree order. Single-owner examples are
  handled with vectorized numpy; multi-owner examples are handled sequentially.
  This satisfies the same formal guarantee as Algorithm 2 of [1] but may differ
  in tie-breaking.

  References: [1] https://arxiv.org/abs/2503.03622

  Args:
    attribution: Multi-owner attribution data.
    contribution_bound: Maximum appearances per user (k in [1]).

  Returns:
    1D array of selected example IDs.
  """
  k = contribution_bound
  if k <= 0 or attribution.num_edges == 0:
    return np.array([], dtype=np.int32)

  m = attribution.num_users
  degree, indptr, sorted_users = attribution._build_csr()  # pylint: disable=protected-access
  user_counts = np.zeros(m, dtype=np.int32)

  # Phase 1: vectorized single-owner examples.
  single_examples = np.where(degree == 1)[0]
  if single_examples.size:
    single_users = sorted_users[indptr[single_examples]]
    # Stable argsort by user preserves example-ID order within each user.
    order = np.argsort(single_users, kind='stable')
    s_users = single_users[order]
    s_examples = single_examples[order]
    # Compute within-user rank to select the first k per user.
    changes = np.empty(len(s_users), dtype=bool)
    changes[0] = True
    np.not_equal(s_users[1:], s_users[:-1], out=changes[1:])
    group_starts = np.where(changes)[0]
    group_sizes = np.diff(np.append(group_starts, len(s_users)))
    rank = np.arange(len(s_users)) - np.repeat(group_starts, group_sizes)
    keep = rank < k
    selected_single = s_examples[keep]
    np.add.at(user_counts, s_users[keep], 1)
  else:
    selected_single = np.array([], dtype=np.int64)

  # Phase 2: sequential multi-owner examples, sorted by degree.
  multi_examples = np.where(degree > 1)[0]
  multi_examples = multi_examples[
      np.argsort(degree[multi_examples], kind='stable')
  ]
  selected_multi = []
  for ex in multi_examples.tolist():
    s, e = int(indptr[ex]), int(indptr[ex + 1])
    # Fast path for the common degree-2 case (scalar indexing, no slice).
    if e - s == 2:
      u0, u1 = int(sorted_users[s]), int(sorted_users[s + 1])
      if user_counts[u0] < k and user_counts[u1] < k:
        selected_multi.append(ex)
        user_counts[u0] += 1
        user_counts[u1] += 1
    else:
      users = sorted_users[s:e]
      if np.all(user_counts[users] < k):
        selected_multi.append(ex)
        np.add.at(user_counts, users, 1)

  return np.concatenate([
      selected_single,
      np.array(selected_multi, dtype=np.int64),
  ])


@dataclasses.dataclass(frozen=True)
class MultiOwnerMinSepSampling(batch_selection.BatchSelectionStrategy):
  """Batch selection with b-min-sep for multi-owner attributed data.

  Greedily fills up to ``iterations * max_batch_size`` slots such that for
  each user, no two batches within a window of ``min_sep`` consecutive batches
  both contain an example attributed to that user. Inspired by Algorithm 4
  of [1].

  The implied per-user contribution bound is ``ceil(iterations / min_sep)``.

  References: [1] https://arxiv.org/abs/2503.03622

  Attributes:
    attribution: Multi-owner attribution data for the dataset.
    max_batch_size: Maximum number of examples per batch (B in [1]). Batches may
      be smaller if the greedy algorithm cannot fill all slots.
    iterations: Total number of batches to produce (T in [1]).
    min_sep: Minimum separation between batches containing examples attributed
      to the same user (b in [1]).
  """

  attribution: MultiOwnerGraph
  max_batch_size: int
  iterations: int
  min_sep: int

  def __post_init__(self):
    if self.min_sep <= 0:
      raise ValueError('min_sep must be positive.')
    if self.iterations < 0:
      raise ValueError('iterations must be non-negative.')
    if self.max_batch_size <= 0:
      raise ValueError('max_batch_size must be positive.')

  def batch_iterator(
      self, num_examples: int, rng: batch_selection.RngType = None
  ) -> Iterator[np.ndarray]:
    """Yields batches satisfying the b-min-sep property.

    Args:
      num_examples: Unused; kept for API compatibility.
      rng: Unused; the algorithm is deterministic.

    Yields:
      1D arrays of example indices, each of length at most ``max_batch_size``.
      Exactly ``iterations`` batches are yielded; batches may be shorter than
      ``max_batch_size`` if the greedy algorithm cannot fill all slots.
    """
    del num_examples, rng
    n = self.attribution.num_examples
    b = self.min_sep
    bs = self.max_batch_size
    total_slots = bs * self.iterations

    if n == 0 or total_slots == 0:
      for _ in range(self.iterations):
        yield np.array([], dtype=np.int32)
      return

    degree, indptr, sorted_users = self.attribution._build_csr()  # pylint: disable=protected-access
    example_order = np.argsort(degree, kind='stable')
    m = self.attribution.num_users

    # last_batch[u] = last batch index in which user u appeared.
    last_batch = np.full(m, -b, dtype=np.int32)
    slots: list[int] = []

    while len(slots) < total_slots:
      progress = False
      for ex in example_order.tolist():
        if len(slots) >= total_slots:
          break
        current_batch = len(slots) // bs
        users = _users_of(indptr, sorted_users, ex)
        if np.all(current_batch - last_batch[users] >= b):
          slots.append(ex)
          last_batch[users] = current_batch
          progress = True
      if not progress:
        break

    # Split into batches (final batches may be undersized).
    result = (
        np.array(slots, dtype=np.int32)
        if slots
        else np.array([], dtype=np.int32)
    )
    for t in range(self.iterations):
      yield result[t * bs : (t + 1) * bs]
