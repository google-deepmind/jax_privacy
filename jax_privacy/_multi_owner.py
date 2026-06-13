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

from collections.abc import Mapping, Sequence
import dataclasses
import functools

from jax_privacy import _validate
import numpy as np


@dataclasses.dataclass(frozen=True)
class MultiOwnerGraph:
  """Bipartite edge list of (example, user) attribution pairs.

  Entry ``(example_ids[i], user_ids[i])`` means that example
  ``example_ids[i]`` is attributed to user ``user_ids[i]``.

  The attribution graph is considered **public information**: it is not
  protected by differential privacy and may be used freely by the batch
  selection algorithm.

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
    validate: If ``False``, skip edge-list validation. Use for large graphs.
  """

  example_ids: np.ndarray
  user_ids: np.ndarray
  validate: bool = True

  def __post_init__(self):
    if self.validate:
      _validate.multi_owner(self.example_ids, self.user_ids)

  @property
  def num_examples(self) -> int:
    """Number of distinct examples (``max(example_ids) + 1``)."""
    return int(self.example_ids.max()) + 1

  @property
  def num_users(self) -> int:
    """Number of distinct users (``max(user_ids) + 1``)."""
    return int(self.user_ids.max()) + 1

  @property
  def num_edges(self) -> int:
    """Total number of (example, user) attribution edges."""
    return len(self.example_ids)

  @classmethod
  def from_owners_per_example(
      cls, owners_per_example: Sequence[np.typing.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a sequence of user sets, one per example.

    Example IDs are assigned sequentially (0, 1, ...). User IDs are
    renumbered to ``[0, num_users)``.

    Args:
      owners_per_example: ``owners_per_example[i]`` is the array-like of user
        IDs attributed to example ``i``. Every example must have at least one
        owner.

    Returns:
      A new MultiOwnerGraph instance.

    Raises:
      ValueError: If any example has zero owners.
    """
    arrays = [np.asarray(o, dtype=np.int64) for o in owners_per_example]
    sizes = np.array([len(a) for a in arrays])
    if len(sizes) > 0 and int(sizes.min()) == 0:
      raise ValueError(
          'Every example must have at least one owner. '
          f'Example(s) at index {np.where(sizes == 0)[0].tolist()} have none.'
      )
    example_ids = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    user_ids = np.concatenate(arrays)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  @classmethod
  def from_user_to_examples(
      cls, user_to_examples: Mapping[int, np.typing.ArrayLike]
  ) -> MultiOwnerGraph:
    """Constructs from a mapping of user IDs to their example sets.

    Both example and user IDs are renumbered to contiguous ranges.
    Users mapped to empty example lists are silently skipped.

    Args:
      user_to_examples: ``user_to_examples[u]`` is the array-like of example IDs
        attributed to user ``u``.

    Returns:
      A new MultiOwnerGraph instance with renumbered IDs.
    """
    ex_arrays, u_arrays = [], []
    for uid, examples in user_to_examples.items():
      ex_arr = np.asarray(examples, dtype=np.int64)
      if len(ex_arr) == 0:
        continue
      ex_arrays.append(ex_arr)
      u_arrays.append(np.full(len(ex_arr), uid, dtype=np.int64))
    if not ex_arrays:
      return cls(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    example_ids = np.concatenate(ex_arrays)
    user_ids = np.concatenate(u_arrays)
    _, example_ids = np.unique(example_ids, return_inverse=True)
    _, user_ids = np.unique(user_ids, return_inverse=True)
    return cls(example_ids=example_ids, user_ids=user_ids)

  @functools.cached_property
  def csr(self) -> _ExampleUserCSR:
    """Compressed sparse row view for per-example user lookup."""
    n = self.num_examples
    degree = np.bincount(self.example_ids, minlength=n)
    order = np.argsort(self.example_ids, kind='stable')
    users = self.user_ids[order]
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    return _ExampleUserCSR(degree=degree, indptr=indptr, sorted_users=users)


@dataclasses.dataclass(frozen=True)
class _ExampleUserCSR:
  """Compressed sparse row layout indexing users by example.

  Stores the user adjacency list for each example in a flat array
  (``sorted_users``) with ``indptr`` marking where each example's users
  begin and end, i.e. the users of example ``e`` are
  ``sorted_users[indptr[e]:indptr[e+1]]``.

  Attributes:
    degree: Number of users per example, shape ``(num_examples,)``.
    indptr: Index pointers into ``sorted_users``, shape ``(num_examples + 1,)``.
    sorted_users: User IDs ordered by example.
  """

  degree: np.ndarray
  indptr: np.ndarray
  sorted_users: np.ndarray

  def users_of(self, ex: int) -> np.ndarray:
    """Returns the user IDs attributed to example ``ex``."""
    return self.sorted_users[self.indptr[ex] : self.indptr[ex + 1]]


def greedy_contribution_bound(
    attribution: MultiOwnerGraph,
    *,
    max_degree: int = 8,
) -> np.ndarray:
  """Selects examples so each user appears at most once.

  Specializes Algorithm 2 of [1] to k=1, processing examples in ascending
  degree order.  Because each user is attributed to at most one selected
  example, the result induces a user-to-example mapping that can be composed
  with any example-level batch selection strategy.


  References: [1] https://arxiv.org/abs/2503.03622

  Args:
    attribution: Multi-owner attribution data.
    max_degree: Maximum number of owners per example to consider.  Must be at
      least 1.

  Returns:
    1D array of selected example IDs.

  Raises:
    ValueError: If ``max_degree < 1``.
  """
  if max_degree < 1:
    raise ValueError(f'max_degree must be >= 1, got {max_degree}.')
  n_ex = attribution.num_examples
  degree = attribution.csr.degree
  ex_ids = attribution.example_ids
  u_ids = attribution.user_ids

  # Priority key: (degree, example_id).  Lower selects first.
  priority = degree[ex_ids].astype(np.int64) * n_ex + ex_ids
  user_active = np.ones(attribution.num_users, dtype=bool)
  best = np.empty(attribution.num_users, dtype=np.int64)
  selected = []
  low_degree = degree <= max_degree

  # Round-based vectorized greedy: each round, every unclaimed user votes for
  # its lowest-priority example, and examples with unanimous votes are selected.
  for _ in range(max_degree):
    active_counts = np.bincount(ex_ids[user_active[u_ids]], minlength=n_ex)
    selectable = (active_counts == degree) & low_degree
    active = selectable[ex_ids]
    ae, au, ap = ex_ids[active], u_ids[active], priority[active]

    # Each user votes for their lowest-priority active example.
    best.fill(np.iinfo(np.int64).max)
    np.minimum.at(best, au, ap)

    # An example is selected iff every one of its users voted for it.
    vote_count = np.bincount(ae[best[au] % n_ex == ae], minlength=n_ex)
    new = np.where(vote_count == degree)[0]

    new_edge_mask = np.isin(ex_ids, new)
    user_active[u_ids[new_edge_mask]] = False
    selected.append(new)

  return np.concatenate(selected)
