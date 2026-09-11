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

"""Fixed-participation batch selection strategies for Monte Carlo accounting.

Two ``jax_privacy.batch_selection.BatchSelectionStrategy`` variants in which
every example participates in exactly ``num_participations`` (K) of the
``iterations`` (T) steps, with consecutive participations at least a minimum
separation apart. Both reduce to existing core strategies in limiting cases.

- ``NestedRandomAllocation``: allocate each example to one of ``cycle_length``
  (b) bins ``i``, then random-allocate K of that bin's iterations
  ``{i, i + b, i + 2b, ...}``. Because a bin's iterations are exactly b apart,
  every gap is a multiple of b and the minimum separation is automatic. Reduces
  to ``RandomAllocationSampling`` when ``cycle_length == 1`` and to
  ``BallsInBinsSampling`` when ``num_participations == 1``.

- ``BalancedMinSep``: fixed-K b-min-sep sampling. With ``random_rotation=False``
  the two ends of training are over-represented. With ``random_rotation=True``
  the K points are placed on a length-T cycle (all gaps ``>= min_sep``) and
  given
  a uniform random rotation, which makes the per-iteration marginal -- and hence
  the expected batch size -- exactly ``K / T`` everywhere. When ``min_sep == 1``
  both settings reduce to ``RandomAllocationSampling``.

Both strategies accept an optional ``sampling_prob``: after the fixed-K
participation set is drawn, each selected participation is independently kept
with that probability (Poisson subsampling), so an example participates in a
``Binomial(K, sampling_prob)`` number of iterations. The default of 1.0 keeps
every participation and recovers the fixed-K behaviour described above. With
``NestedRandomAllocation`` and ``num_participations == iterations //
cycle_length`` (every candidate selected), this recovers cyclic Poisson
sampling with ``partition_type=INDEPENDENT``.
"""

import dataclasses
import enum
import math
from typing import Iterator

from jax_privacy import batch_selection
import numpy as np

__all__ = [
    'BalancedMinSep',
    'NestedRandomAllocation',
    'Strategy',
    'build_strategy',
]


def _random_compositions(
    rng: np.random.Generator,
    num_examples: int,
    total: int,
    num_parts: int,
) -> np.ndarray:
  """Returns uniform compositions of ``total`` into ``num_parts`` parts.

  Each row is an independent uniform draw from the nonnegative integer vectors
  of
  length ``num_parts`` that sum to ``total`` (stars and bars): choose
  ``num_parts - 1`` distinct bars in ``[0, total + num_parts - 1)``; the gaps
  between the sorted bars (minus one) are the parts.

  Args:
    rng: NumPy random generator.
    num_examples: The number of independent compositions (rows) to draw.
    total: The value ``T`` each composition sums to.
    num_parts: The number of parts each composition is split into.

  Returns:
    Int array of shape ``(num_examples, num_parts)`` with rows summing to
    ``total``.
  """
  n = total + num_parts - 1
  bars = np.sort(
      rng.random((num_examples, n)).argsort(axis=1)[:, : num_parts - 1], axis=1
  )
  lo = np.full((num_examples, 1), -1)
  hi = np.full((num_examples, 1), n)
  return np.diff(np.concatenate([lo, bars, hi], axis=1), axis=1) - 1


@dataclasses.dataclass(frozen=True)
class NestedRandomAllocation(batch_selection.BatchSelectionStrategy):
  """Random allocation within a randomly chosen residue class.

  Each example is assigned a bin ``i`` uniformly from ``{0, ..., b - 1}`` and
  then participates in exactly ``num_participations`` of the candidate
  iterations ``{i, i + b, i + 2b, ...} n [0, T)``, chosen by random allocation.
  Consecutive participations are therefore always at least ``cycle_length``
  iterations apart.

  When ``sampling_prob < 1`` each selected participation is then kept
  independently with that probability, so an example participates in
  ``Binomial(num_participations, sampling_prob)`` iterations. Setting
  ``num_participations == iterations // cycle_length`` (every candidate
  selected) recovers ``CyclicPoissonSampling`` with
  ``partition_type=INDEPENDENT`` and this ``sampling_prob``.

  Attributes:
    cycle_length: The number of bins ``b``, equivalently the minimum separation
      between consecutive participations of the same example.
    num_participations: The number of iterations each example is allocated to
      before subsampling.
    iterations: The total number of iterations / batches to generate.
    sampling_prob: Probability of keeping each allocated participation. The
      default of 1.0 keeps all of them (no subsampling).
  """

  cycle_length: int
  num_participations: int
  iterations: int
  sampling_prob: float = 1.0

  def __post_init__(self):
    # Every bin has at least ``iterations // cycle_length`` candidates.
    assert self.num_participations <= self.iterations // self.cycle_length
    assert 0.0 <= self.sampling_prob <= 1.0

  def batch_iterator(
      self, num_examples: int, rng: batch_selection.RngType = None
  ) -> Iterator[np.ndarray]:
    b = self.cycle_length
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(-num_examples)
    bins = rng.integers(b, size=num_examples)
    # Each example random-allocates its participations among the iterations in
    # its own residue class, streamed exactly like RandomAllocationSampling: an
    # example with `remaining` participations and `candidates_left` candidate
    # iterations left participates with probability remaining / candidates_left.
    remaining = np.full(num_examples, self.num_participations)
    # Bin i holds iterations {i, i + b, ...}, i.e. ceil((T - i) / b) candidates.
    candidates_left = (self.iterations - np.arange(b) + b - 1) // b
    for step in range(self.iterations):
      i = step % b
      probs = np.where(bins == i, remaining / candidates_left[i], 0.0)
      selected = rng.random(num_examples) < probs
      # Poisson-subsample the allocated participations. The allocation state
      # (``remaining``) is decremented by ``selected``, not the kept subset, so
      # the residue-class structure and min separation are preserved.
      if self.sampling_prob < 1.0:
        kept = selected & (rng.random(num_examples) < self.sampling_prob)
      else:
        kept = selected
      yield np.where(kept)[0].astype(dtype)
      remaining -= selected
      candidates_left[i] -= 1


@dataclasses.dataclass(frozen=True)
class BalancedMinSep(batch_selection.BatchSelectionStrategy):
  """Fixed-K b-min-sep sampling, optionally balanced to a uniform marginal.

  Each example participates in exactly ``num_participations`` (K) iterations
  with
  consecutive participations at least ``min_sep`` (b) apart. When
  ``min_sep == 1`` this is exactly ``RandomAllocationSampling`` regardless of
  ``random_rotation``.

  When ``sampling_prob < 1`` each selected participation is then kept
  independently with that probability, so an example participates in
  ``Binomial(num_participations, sampling_prob)`` iterations.

  Attributes:
    min_sep: The minimum separation ``b`` between consecutive participations.
    num_participations: The number of iterations each example is allocated to
      before subsampling.
    iterations: The total number of iterations / batches to generate.
    random_rotation: If True, place the participations on a cycle and apply a
      uniform random rotation so that every iteration has the same expected
      batch size. If False, the two ends of training are over-represented.
    sampling_prob: Probability of keeping each allocated participation. The
      default of 1.0 keeps all of them (no subsampling).
  """

  min_sep: int
  num_participations: int
  iterations: int
  random_rotation: bool = False
  sampling_prob: float = 1.0

  def __post_init__(self):
    k, b, t = self.num_participations, self.min_sep, self.iterations
    if self.random_rotation:
      assert t >= k * b
    else:
      assert t >= (k - 1) * b + 1
    assert 0.0 <= self.sampling_prob <= 1.0

  def batch_iterator(
      self, num_examples: int, rng: batch_selection.RngType = None
  ) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng)
    dtype = np.min_scalar_type(-num_examples)
    if self.random_rotation:
      positions = self._cyclic_positions(num_examples, rng)
    else:
      positions = self._linear_positions(num_examples, rng)
    for step in range(self.iterations):
      selected = np.any(positions == step, axis=1)
      # Poisson-subsample: keep each selected participation independently. The
      # positions are fixed, so the min-sep structure is preserved.
      if self.sampling_prob < 1.0:
        kept = selected & (rng.random(num_examples) < self.sampling_prob)
      else:
        kept = selected
      yield np.where(kept)[0].astype(dtype)

  def _linear_positions(
      self, num_examples: int, rng: np.random.Generator
  ) -> np.ndarray:
    """K positions in [0, T); interior gaps >= min_sep, ends free."""
    k, b, t = self.num_participations, self.min_sep, self.iterations
    # K - 1 interior gaps of size >= b plus free space at both ends: distribute
    # the leftover T - 1 - (K - 1) b over K + 1 parts, keep the first K, and
    # offset by b per step. Free ends over-represent the start and end of
    # training.
    parts = _random_compositions(rng, num_examples, t - 1 - (k - 1) * b, k + 1)
    return np.cumsum(parts[:, :k], axis=1) + np.arange(k) * b

  def _cyclic_positions(
      self, num_examples: int, rng: np.random.Generator
  ) -> np.ndarray:
    """K cyclic points with all gaps >= min_sep, then a uniform rotation."""
    k, b, t = self.num_participations, self.min_sep, self.iterations
    # K cyclic gaps each >= b summing to T: distribute the slack T - K b over K
    # parts, add b to each; the rotation flattens the marginal to K / T.
    parts = _random_compositions(rng, num_examples, t - k * b, k)
    gaps = parts + b
    starts = np.cumsum(gaps, axis=1) - gaps
    rotation = rng.integers(t, size=(num_examples, 1))
    return (starts + rotation) % t


class Strategy(enum.Enum):
  """Named batch-selection strategies constructible by ``build_strategy``.

  Members:
    BALLS_IN_BINS: Each example is assigned a residue class and participates in
      every one of its candidate iterations (deterministic, min separation
      ``bands``). Requires ``expected_participations == iterations // bands``.
    RA_CYCLIC_POISSON: Like balls-in-bins with random group assignment, but each
      candidate participation is independently kept with a probability chosen so
      the expected count is ``expected_participations`` (cyclic Poisson sampling
      with independent partition).
    NESTED_RANDOM_ALLOCATION: Random group assignment followed by random
      allocation of a fixed subset of the residue class' candidate iterations.
    BALANCED_MIN_SEP: Fixed-K ``bands``-min-sep sampling with the two ends of
      training over-represented.
    BALANCED_MIN_SEP_ROTATED: Fixed-K ``bands``-min-sep sampling on a cycle with
      a uniform random rotation, giving a uniform per-iteration marginal.
    MIN_SEP_POISSON: Poisson ``bands``-min-sep sampling, routed to the core JAX
      Privacy ``BMinSepSampling`` strategy. Every example is eligible in each
      iteration unless it participated in the previous ``bands`` - 1 iterations.
  """

  BALLS_IN_BINS = 'balls_in_bins'
  RA_CYCLIC_POISSON = 'ra_cyclic_poisson'
  NESTED_RANDOM_ALLOCATION = 'nested_random_allocation'
  BALANCED_MIN_SEP = 'balanced_min_sep'
  BALANCED_MIN_SEP_ROTATED = 'balanced_min_sep_rotated'
  MIN_SEP_POISSON = 'min_sep_poisson'


def build_strategy(
    strategy: Strategy | str,
    *,
    iterations: int,
    expected_participations: float,
    bands: int,
) -> batch_selection.BatchSelectionStrategy:
  """Builds a batch-selection strategy from high-level primitives.

  Every example participates in ``expected_participations`` iterations in
  expectation, with consecutive participations at least ``bands`` apart. For the
  fixed-K strategies a non-integer expectation is realized by allocating
  ``ceil(expected_participations)`` participations and Poisson-subsampling each
  with probability ``expected_participations / ceil(expected_participations)``.

  Args:
    strategy: The strategy to build, as a ``Strategy`` or its string value.
    iterations: The total number of iterations / batches ``T``.
    expected_participations: The expected number of participations per example.
    bands: The minimum separation ``b`` between consecutive participations.

  Returns:
    The constructed batch-selection strategy.

  Raises:
    ValueError: If the arguments are non-positive or inconsistent with the
      chosen strategy.
  """
  strategy = Strategy(strategy)
  t, e, b = iterations, expected_participations, bands
  if t <= 0 or b <= 0:
    raise ValueError(f'iterations and bands must be positive; got {t=}, {b=}.')
  if e <= 0:
    raise ValueError(f'expected_participations must be positive; got {e}.')
  candidates_per_class = t // b  # Size of the residue class {i, i + b, ...}.

  if strategy is Strategy.BALLS_IN_BINS:
    if e != candidates_per_class:
      raise ValueError(f'BALLS_IN_BINS requires E == {candidates_per_class}.')
    # Expressed as NestedRandomAllocation, not the core BallsInBinsSampling, to
    # reuse the JAX sampling + scoring code that runs on GPUs.
    return NestedRandomAllocation(
        cycle_length=b, num_participations=candidates_per_class, iterations=t
    )

  if strategy is Strategy.RA_CYCLIC_POISSON:
    if e > candidates_per_class:
      raise ValueError(f'RA_CYCLIC_POISSON needs E <= {candidates_per_class}.')
    # Expressed as NestedRandomAllocation, not the core CyclicPoissonSampling,
    # to reuse the JAX sampling + scoring code that runs on GPUs.
    return NestedRandomAllocation(
        cycle_length=b,
        num_participations=candidates_per_class,
        iterations=t,
        sampling_prob=e / candidates_per_class,
    )

  if strategy is Strategy.MIN_SEP_POISSON:
    # E = T / (b - 1 + 1 / p) with a warm start, so p = E / (T - E * (b - 1)).
    if e * b > t:
      raise ValueError(f'MIN_SEP_POISSON requires E * bands <= {t}.')
    return batch_selection.BMinSepSampling(
        sampling_prob=e / (t - e * (b - 1)), iterations=t, min_sep=b
    )

  # Remaining strategies use a fixed number of participations K = ceil(E) and
  # subsample to match a fractional expected participation count.
  k = math.ceil(e)
  sampling_prob = e / k
  if strategy is Strategy.NESTED_RANDOM_ALLOCATION:
    return NestedRandomAllocation(
        cycle_length=b,
        num_participations=k,
        iterations=t,
        sampling_prob=sampling_prob,
    )
  if strategy in (Strategy.BALANCED_MIN_SEP, Strategy.BALANCED_MIN_SEP_ROTATED):
    return BalancedMinSep(
        min_sep=b,
        num_participations=k,
        iterations=t,
        random_rotation=strategy is Strategy.BALANCED_MIN_SEP_ROTATED,
        sampling_prob=sampling_prob,
    )
  raise ValueError(f'Unsupported strategy: {strategy}.')
