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

"""Monte Carlo privacy-loss histogram: sampling, scoring, and discretization.

This is the *privacy-critical* module and warrants close scrutiny. It draws
dominating-pair observations, scores each observation's privacy loss, and rounds
the losses onto a grid to form a discretized privacy-loss distribution (PLD).
The microbatched accumulation loop it relies on (``_accumulate_bincounts``) is
privacy-agnostic performance plumbing: a bug there would change speed, memory,
or Monte Carlo variance, never the validity of the discretized bound.

Two histograms are accumulated, one per adjacency direction:

- ``counts_pos``: observations ``y ~ P`` (sensitive example present), scored by
  the privacy loss ``L(y) = log[P(y) / Q(y)]``.
- ``counts_neg``: observations ``y ~ Q`` (absent), scored by ``-L(y)``.

Discretization follows ``dp_accounting``'s PLD convention: losses are rounded
UP to a grid point, so each stored loss is an over-estimate of the true loss.
Downstream hockey-stick accounting on this histogram therefore over-estimates
``delta`` (up to Monte Carlo error); it is a Monte Carlo estimate, not a
high-probability upper bound (formal calibration is handled elsewhere).

Run under ``jax.enable_x64`` -- the log-sum-exp scoring needs float64.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
from research.monte_carlo import batch_selection
from research.monte_carlo import sample_generation
from research.monte_carlo import scoring
import numpy as np

__all__ = ['PrivacyLossHistogram', 'privacy_loss_histogram']

# The strategies whose scoring is supported by `scoring.compute_privacy_loss`.
_Strategy = (
    batch_selection.BalancedMinSep | batch_selection.NestedRandomAllocation
)


def _make_pattern_sampler(strategy: _Strategy):
  """Returns ``key -> (positions, poisson_mask)`` for ``strategy``."""
  if isinstance(strategy, batch_selection.NestedRandomAllocation):
    return lambda key: sample_generation.sample_nested_random_allocation(
        key,
        strategy.cycle_length,
        strategy.num_participations,
        strategy.iterations,
        sampling_prob=strategy.sampling_prob,
    )
  if isinstance(strategy, batch_selection.BalancedMinSep):
    return lambda key: sample_generation.sample_balanced_min_sep(
        key,
        strategy.min_sep,
        strategy.num_participations,
        strategy.iterations,
        random_rotation=strategy.random_rotation,
        sampling_prob=strategy.sampling_prob,
    )
  raise ValueError(f'Unsupported strategy: {type(strategy).__name__}')


def _round_up_to_bin(
    loss: jax.Array, grid_lo: float, grid_step: float, n_bins: int
) -> jax.Array:
  """Rounds losses UP onto the grid, as bin indices in ``[0, n_bins + 1]``."""
  # Bin i in 0..n_bins is the grid point grid_lo + i*grid_step (an over-estimate
  # of the true loss); losses <= grid_lo clamp to bin 0 and losses > grid_hi map
  # to the +inf overflow bin at index n_bins + 1.
  idx = jnp.ceil((loss - grid_lo) / grid_step).astype(jnp.int32)
  return jnp.clip(idx, 0, n_bins + 1)


@dataclasses.dataclass(frozen=True)
class PrivacyLossHistogram:
  """Privacy losses rounded UP onto a fixed grid (a pessimistic PLD).

  The grid covers ``[grid_lo, grid_hi]`` in ``grid_step`` increments. The count
  arrays have ``n_bins + 2`` entries: index ``i`` in ``0 .. n_bins`` holds the
  mass rounded up to grid point ``grid_lo + i * grid_step``, and index
  ``n_bins + 1`` holds the ``+inf`` overflow mass (losses above ``grid_hi``).

  Attributes:
    counts_pos: Histogram of ``L(y)`` for ``y ~ P``. Shape ``(n_bins + 2,)``.
    counts_neg: Histogram of ``-L(y)`` for ``y ~ Q``. Shape ``(n_bins + 2,)``.
    grid_lo: Lowest grid point (bin 0).
    grid_hi: Highest grid point (bin ``n_bins``).
    grid_step: Spacing between grid points.
  """

  counts_pos: np.ndarray
  counts_neg: np.ndarray
  grid_lo: float
  grid_hi: float
  grid_step: float

  @property
  def n_bins(self) -> int:
    """The number of in-range grid points minus one."""
    return len(self.counts_pos) - 2

  @property
  def num_samples(self) -> int:
    """The number of samples per direction."""
    return int(np.sum(self.counts_pos))

  @property
  def overflow_fraction(self) -> float:
    """Largest fraction of either direction's mass in the overflow bin.

    A nonzero value means the grid's upper bound is too low: those samples land
    above ``grid_hi`` and are treated pessimistically (as ``+inf``) by any
    downstream accounting, so widen ``grid_hi`` for a tighter estimate.
    """
    pos = self.counts_pos[-1] / max(1, self.num_samples)
    neg = self.counts_neg[-1] / max(1, self.num_samples)
    return float(max(pos, neg))


def _accumulate_bincounts(
    per_sample, num_samples, num_bins, *, microbatch_size
):
  """Sums per-sample bin indices into histograms of length ``num_bins``.

  Privacy-agnostic performance plumbing: a ``fori_loop`` over microbatches
  derives sample indices from the loop counter, so no length-``num_samples``
  buffer is materialised and peak memory is ``O(num_bins)`` plus one microbatch.

  Args:
    per_sample: Maps a scalar sample index to a pytree of scalar bin indices in
      ``[0, num_bins)``.
    num_samples: Total number of samples to accumulate; must be a multiple of
      ``microbatch_size``.
    num_bins: Length of each output count vector.
    microbatch_size: Samples processed per loop iteration; must divide
      ``num_samples``.

  Returns:
    A pytree matching ``per_sample``'s output, each leaf a length-``num_bins``
    count vector.
  """

  def block(start):
    indices = start + jnp.arange(microbatch_size, dtype=jnp.uint32)
    bins = jax.vmap(per_sample)(indices)
    return jax.tree.map(lambda b: jnp.bincount(b, length=num_bins), bins)

  def body(i, carry):
    return jax.tree.map(jnp.add, carry, block(i * microbatch_size))

  num_microbatches = num_samples // microbatch_size
  return jax.lax.fori_loop(1, num_microbatches, body, block(0))


@jax.jit(
    static_argnums=(0,),
    static_argnames=(
        'num_samples',
        'n_bins',
        'grid_lo',
        'grid_step',
        'microbatch_size',
    ),
)
def _sample_losses_and_bin(
    strategy: _Strategy,
    c_col: jax.Array,
    noise_multiplier: float,
    key: jax.Array,
    *,
    num_samples: int,
    n_bins: int,
    grid_lo: float,
    grid_step: float,
    microbatch_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Jitted sampling + scoring + histogram accumulation (see public wrapper)."""
  iterations = strategy.iterations
  sample_pattern = _make_pattern_sampler(strategy)

  def per_sample(index: jax.Array) -> tuple[jax.Array, jax.Array]:
    sample_key = jax.random.fold_in(key, index)
    pattern_key, pos_key, neg_key = jax.random.split(sample_key, 3)
    # The sampler returns the participation positions and a Poisson keep-mask
    # (all-True unless the strategy subsamples); honour the mask when scoring.
    pattern, keep = sample_pattern(pattern_key)
    mode = sample_generation.banded_c_times_pattern(
        c_col, pattern, iterations, keep
    )
    noise_pos = jax.random.normal(pos_key, (iterations,), dtype=c_col.dtype)
    noise_neg = jax.random.normal(neg_key, (iterations,), dtype=c_col.dtype)
    # Positive: y ~ P (mode + noise); negative: y ~ Q (noise), loss negated.
    loss_pos = scoring.compute_privacy_loss(
        strategy, mode + noise_multiplier * noise_pos, noise_multiplier, c_col
    )
    loss_neg = -scoring.compute_privacy_loss(
        strategy, noise_multiplier * noise_neg, noise_multiplier, c_col
    )
    return (
        _round_up_to_bin(loss_pos, grid_lo, grid_step, n_bins),
        _round_up_to_bin(loss_neg, grid_lo, grid_step, n_bins),
    )

  return _accumulate_bincounts(
      per_sample, num_samples, n_bins + 2, microbatch_size=microbatch_size
  )


def privacy_loss_histogram(
    strategy: _Strategy,
    c_col: jax.Array,
    noise_multiplier: float,
    key: jax.Array,
    *,
    num_samples: int,
    microbatch_size: int,
    grid_lo: float = -20.0,
    grid_hi: float = 30.0,
    grid_step: float = 1e-3,
) -> PrivacyLossHistogram:
  """Accumulates the privacy-loss histogram for a strategy and noise level.

  Draws ``num_samples`` dominating-pair observations in each direction (rounded
  up to a whole number of microbatches), scores them with
  ``scoring.compute_privacy_loss``, and rounds the losses up onto the
  grid ``[grid_lo, grid_hi]`` in steps of ``grid_step``. All work runs in a
  single jitted ``fori_loop``, so peak memory scales with the number of bins
  plus one microbatch of activations, not with ``num_samples``; a single call
  therefore handles arbitrarily many samples.

  Args:
    strategy: The batch selection strategy. Supported: ``BalancedMinSep`` with
      ``random_rotation=False`` and ``NestedRandomAllocation``.
    c_col: The nonzero first-column band of the Toeplitz matrix ``C``, with
      ``len(c_col) <= b`` so participation bands do not overlap.
    noise_multiplier: Standard deviation ``sigma`` of the Gaussian noise.
    key: JAX PRNG key.
    num_samples: Minimum number of samples per direction; rounded up to a whole
      multiple of ``microbatch_size``.
    microbatch_size: Samples processed per loop step (the memory/speed knob).
      Larger values run faster but use more device memory; ``num_samples`` is
      rounded up to a multiple of this so every microbatch is full.
    grid_lo: Lowest grid point.
    grid_hi: Highest grid point.
    grid_step: Spacing between grid points.

  Returns:
    A ``PrivacyLossHistogram``.
  """
  n_bins = int(round((grid_hi - grid_lo) / grid_step))
  if microbatch_size <= 0:
    raise ValueError(
        f'microbatch_size must be positive; got {microbatch_size}.'
    )
  microbatch_size = min(microbatch_size, num_samples)
  # Round up so every microbatch is full; treats num_samples as a minimum.
  num_samples = math.ceil(num_samples / microbatch_size) * microbatch_size

  counts_pos, counts_neg = _sample_losses_and_bin(
      strategy,
      jnp.asarray(c_col),
      noise_multiplier,
      key,
      num_samples=num_samples,
      n_bins=n_bins,
      grid_lo=grid_lo,
      grid_step=grid_step,
      microbatch_size=microbatch_size,
  )
  return PrivacyLossHistogram(
      counts_pos=np.asarray(counts_pos),
      counts_neg=np.asarray(counts_neg),
      grid_lo=grid_lo,
      grid_hi=grid_hi,
      grid_step=grid_step,
  )
