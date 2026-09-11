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

"""Pure-JAX privacy-loss scoring via dynamic programming.

Given an observation ``y = mode + sigma * z`` from the dominating pair of a
fixed-participation strategy (``mode`` encodes which iterations the sensitive
example participated in, scattered through the banded Toeplitz matrix ``C``),
``compute_privacy_loss`` returns the privacy loss

    L(y) = log [ p(y | example present) / p(y | example absent) ],

marginalising over the example's (random) participation pattern with a dynamic
program. Drawing many positive/negative samples and feeding their losses to a
hockey-stick accountant yields Monte Carlo (epsilon, delta) estimates.

Both strategies assume ``len(c_col) <= b`` (the separation), so the bands from
distinct participations never overlap and the per-iteration Gaussian
log-likelihood ratios are additive:

    llr[i] = (2 * <c_col, y[i : i + len(c_col)]> - ||c_col_trunc||^2) / (2
    sigma^2)

Every function operates on a single length-``iterations`` sample and returns a
scalar; draw a batch by mapping over the sample axis, e.g.
``jax.vmap(compute_privacy_loss, in_axes=(None, 1, None, None))``.
"""

import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax_privacy import batch_selection as core_batch_selection
from research.monte_carlo import batch_selection

__all__ = ['compute_privacy_loss']

# Unroll factor for the scoring recurrence (see ``_ksubset_log_mgf``). Feeding
# the loop as a ``lax.scan`` unrolled by this much lets XLA overlap the
# stride-independent buffer updates and hide ``logaddexp`` latency, giving ~1.7x
# on GPU for strided (``b > 1``) kernels; it is neutral for stride-1 draws and
# ~2% slower on TPU. Numerically identical to a plain loop.
_SCAN_UNROLL = 4


def _sliding_dot_products(c_col: jax.Array, sample: jax.Array) -> jax.Array:
  """Returns ``dot[i] = <c_col, sample[i : i + b]>`` (zero-padded)."""
  full = jsp.signal.fftconvolve(sample, c_col[::-1], mode='full')
  return full[c_col.shape[0] - 1 :]


def _boundary_aware_sq_norms(c_col: jax.Array, t: int) -> jax.Array:
  """Returns ``||c_col[:min(b, t - i)]||^2`` for ``i`` in ``[0, t)``."""
  sq = jnp.ones(t, dtype=c_col.dtype) * jnp.sum(c_col**2)
  if c_col.shape[0] > 1:
    # The last b - 1 rows see a truncated band, so their norm shrinks.
    sq = sq.at[-c_col.shape[0] + 1 :].set(jnp.cumsum(c_col**2)[:-1][::-1])
  return sq


def _log_likelihood_ratios(
    c_col: jax.Array, sample: jax.Array, noise_multiplier: float
) -> jax.Array:
  """Per-iteration Gaussian LLRs for a participation at each iteration."""
  dots = _sliding_dot_products(c_col, sample)
  sq_norms = _boundary_aware_sq_norms(c_col, sample.shape[0])
  return (2.0 * dots - sq_norms) / (2.0 * noise_multiplier**2)


def _ksubset_log_mgf(
    p_part: jax.Array, llrs: jax.Array, k: int, stride: int
) -> jax.Array:
  """``log E_A[exp(sum_{j in A} llrs[j])]`` for a sequential random K-subset A.

  Both supported strategies draw the example's participation set by the same
  sequential process, so they share this recurrence. Sweeping positions
  backwards with ``r`` participations still to place, the example either

    * participates at the current position -- with probability ``p_part[pos,
    r-1]``
      -- adding ``llrs[pos]`` and jumping ``stride`` positions ahead, or
    * skips it, advancing a single position.

  A rolling buffer of ``stride`` rows suffices because the participate branch
  lands exactly ``stride`` rows ahead (row ``pos`` and row ``pos + stride``
  never
  coexist). For a plain advance-by-one process ``stride == 1`` and the buffer is
  a single row. ``buf[., r]`` holds the log-MGF of choosing ``r`` more
  participations from the suffix; ``buf[., 0]`` stays 0.

  Args:
    p_part: ``(len(llrs), K)`` participation probabilities; ``p_part[pos, r-1]``
      is the chance of participating at ``pos`` with ``r`` left to place.
    llrs: Per-position LLRs of shape ``(len(llrs),)``.
    k: Number of participations K.
    stride: Positions skipped by a participation (``min_sep`` / cycle length).

  Returns:
    Scalar log-MGF ``buf[0, K]``.
  """
  n = llrs.shape[0]
  log_part = jnp.log(p_part)
  # Forced (p_part == 1) or impossible transitions give log = -inf, which is
  # harmless: logaddexp handles it and p_part is a data-independent constant.
  log_skip = jnp.log1p(-p_part)
  buf = jnp.zeros((stride, k + 1), dtype=llrs.dtype)

  def body(buf, loop_idx):
    pos = n - 1 - loop_idx
    skip = log_skip[pos] + buf[(pos + 1) % stride, 1 : k + 1]
    part = log_part[pos] + llrs[pos] + buf[pos % stride, 0:k]
    buf = buf.at[pos % stride, 1 : k + 1].set(jnp.logaddexp(skip, part))
    return buf, None

  # Feeding positions as scanned inputs (rather than a fori_loop counter) keeps
  # the stride-separated updates independent, so unrolling exposes ILP.
  buf, _ = jax.lax.scan(body, buf, jnp.arange(n), unroll=_SCAN_UNROLL)
  return buf[0, k]


def _balanced_min_sep_log_loss(llrs: jax.Array, k: int, b: int) -> jax.Array:
  """Log privacy loss for linear (non-rotated) fixed-K b-min-sep sampling.

  Linear b-min-sep sampling is uniform over the K-subsets of ``[0, T)`` whose
  consecutive elements are at least ``b`` apart. The implied participation
  probability at iteration ``i`` with ``r`` left to place is
  ``r / (T - i - (r-1)(b-1))``, and a participation forces the next ``b-1``
  iterations to be skipped -- i.e. a stride-``b`` sequential K-subset draw.

  Args:
    llrs: Per-iteration LLRs of shape ``(T,)``.
    k: Number of participations K.
    b: Minimum separation.

  Returns:
    Scalar log privacy loss.
  """
  t = llrs.shape[0]
  r_vals = jnp.arange(1, k + 1)
  i_vals = jnp.arange(t)[:, None]
  eligible = jnp.maximum(t - i_vals - (r_vals[None, :] - 1) * (b - 1), 1.0)
  p_part = jnp.minimum(r_vals[None, :] / eligible, 1.0)  # (T, K)
  return _ksubset_log_mgf(p_part, llrs, k, stride=b)


def _suffix_log_counts(llrs, k, b, upper):
  """Per-position log weighted counts of linear min-sep suffix subsets.

  ``cols[pos]`` is ``log sum`` over the min-sep ``(k - 1)``-subsets of
  ``[pos, upper]`` of ``prod_j exp(llrs[j])``; positions above ``upper`` are
  never selected. A rolling buffer of the last ``b`` rows -- each the length-k
  vector of log counts for choosing ``0 .. k - 1`` more positions -- suffices
  because a chosen position lands exactly ``b`` rows ahead.

  Args:
    llrs: Per-iteration LLRs of shape ``(T,)``.
    k: Number of participations K.
    b: Minimum separation.
    upper: Largest position index that may be selected.

  Returns:
    Array of shape ``(T,)`` giving the ``r = k - 1`` log count at each position.
  """
  t = llrs.shape[0]
  neg_inf = jnp.array(-jnp.inf, llrs.dtype)
  zero = jnp.zeros(1, llrs.dtype)
  buf0 = jnp.broadcast_to(jnp.r_[zero, jnp.full(k - 1, neg_inf)], (b, k))

  def body(buf, pos):
    skip = buf[(pos + 1) % b]
    prev = buf[pos % b]  # Row for pos + b, before this position overwrites it.
    take = jnp.where(pos <= upper, llrs[pos] + prev[: k - 1], neg_inf)
    row = jnp.r_[zero, jnp.logaddexp(skip[1:], take)]
    return buf.at[pos % b].set(row), row[k - 1]

  _, cols = jax.lax.scan(body, buf0, jnp.arange(t - 1, -1, -1))
  return cols[::-1]  # scan ran on decreasing positions; index by position.


def _cyclic_log_normalizer(t, k, b):
  """Log count of valid cyclic K-subsets: (t / k) C(t - k(b-1) - 1, k - 1)."""
  n = t - k * (b - 1) - 1
  log_c = math.lgamma(n + 1) - math.lgamma(k) - math.lgamma(n - k + 2)
  return math.log(t) - math.log(k) + log_c


def _balanced_min_sep_cyclic_log_loss(llrs, k, b):
  """Log privacy loss for cyclic (rotated) fixed-K b-min-sep sampling.

  ``random_rotation=True`` draws a uniform K-subset of the cycle ``Z_T`` whose
  K cyclic gaps (including the wrap-around) are all ``>= b``, then rotates it --
  giving a flat ``K / T`` marginal. Conditioning on the minimum selected
  position ``F``, the other ``K - 1`` positions form a linear min-sep
  ``(K - 1)``-subset of ``[F + b, T)`` whose maximum is ``<= T - b + F`` (the
  wrap constraint). That bound is vacuous once ``F >= b - 1``; only the
  ``b - 1`` boundary values need the top ``b - 1 - F`` positions excluded,
  done by re-running the suffix DP with a lowered cutoff. Dividing by the
  number of valid cyclic subsets normalises the uniform law. Costs ``O(b)``
  suffix passes versus one for the linear law.

  Args:
    llrs: Per-iteration LLRs of shape ``(T,)``.
    k: Number of participations K.
    b: Minimum separation (also the cyclic gap bound).

  Returns:
    Scalar log privacy loss.
  """
  t = llrs.shape[0]
  if k == 1:
    # A lone participation always has cyclic gap T >= b, so every position is
    # valid and the law is uniform over all T positions.
    return jsp.special.logsumexp(llrs) - jnp.log(t)
  neg_inf = jnp.array(-jnp.inf, llrs.dtype)
  fb = jnp.arange(t) + b
  # F >= b - 1: wrap bound vacuous, so the unbounded suffix DP applies.
  col = _suffix_log_counts(llrs, k, b, t - 1)
  s = jnp.where(fb < t, col[jnp.minimum(fb, t - 1)], neg_inf)
  if b > 1:
    # F < b - 1: exclude the top b - 1 - F positions via a per-F cutoff.
    fs = jnp.arange(b - 1)
    cols = jax.vmap(lambda u: _suffix_log_counts(llrs, k, b, u))(t - b + fs)
    s = s.at[fs].set(cols[fs, fs + b])
  return jsp.special.logsumexp(llrs + s) - _cyclic_log_normalizer(t, k, b)


def _nested_random_allocation_log_loss(
    llrs: jax.Array, k: int, b: int
) -> jax.Array:
  """Log privacy loss for nested random allocation (residue-class allocation).

  The example first picks a bin ``i`` uniformly from ``{0, ..., b-1}`` and then
  a
  uniform K-subset of that bin's candidate iterations
  ``P_i = {i, i+b, i+2b, ...} n [0, T)``. Because the candidates within a bin
  are
  exactly ``b`` apart (and ``len(c_col) <= b``), their bands are disjoint, so
  the
  per-bin likelihood ratio is a plain random-allocation DP (a stride-1 K-subset
  draw with participation probability ``r / (m_i - c)`` for the ``c``-th
  candidate), and the bins combine by a uniform mixture:

    L(y) = logsumexp_i D_i(y) - log(b),
    D_i(y) = log E_{A ~ unif K-subset of P_i}[ exp(sum_{j in A} llr[j]) ].

  As a sanity check, taking ``K = m_i`` (participate in every candidate)
  recovers
  the balls-in-bins privacy loss
  ``logsumexp_i (sum_{j in P_i} llr[j]) - log(b)``.

  Args:
    llrs: Per-iteration LLRs of shape ``(T,)``.
    k: Number of participations K.
    b: Number of bins / cycle length (also the minimum separation).

  Returns:
    Scalar log privacy loss.
  """
  t = llrs.shape[0]
  m = (t + b - 1) // b  # Max candidates in any residue class.

  # Gather the LLRs of each residue class into (b, m); pad the ragged tail
  # (classes with only m-1 candidates) with -inf so those slots are never
  # chosen.
  positions = jnp.arange(b)[:, None] + jnp.arange(m)[None, :] * b  # (b, m)
  valid = positions < t
  gathered = llrs[jnp.clip(positions, 0, t - 1)]  # (b, m)
  llr_by_class = jnp.where(valid, gathered, -jnp.inf)
  counts = valid.sum(axis=1)  # (b,) number of candidates m_i per class.

  def class_log_loss(llr_c, m_i):
    """Random-allocation (stride-1) DP over one residue class' candidates."""
    r_vals = jnp.arange(1, k + 1)
    c_vals = jnp.arange(m)[:, None]  # (m, 1)
    denom = jnp.maximum(m_i - c_vals, 1.0)
    p_part = jnp.where(
        c_vals < m_i, jnp.minimum(r_vals[None, :] / denom, 1.0), 0.0
    )  # (m, K); padding candidates get p_part = 0 (always skipped).
    return _ksubset_log_mgf(p_part, llr_c, k, stride=1)

  per_class = jax.vmap(class_log_loss)(llr_by_class, counts)  # (b,)
  return jsp.special.logsumexp(per_class) - jnp.log(b)


def compute_privacy_loss(
    strategy: core_batch_selection.BatchSelectionStrategy,
    sample: jax.Array,
    noise_multiplier: float,
    c_col: jax.Array,
) -> jax.Array:
  """Computes the privacy loss of an observed sample under ``strategy``.

  Args:
    strategy: The batch selection strategy. Supported: ``BalancedMinSep``
      (linear or ``random_rotation=True``) and ``NestedRandomAllocation``.
      Either may set ``sampling_prob < 1`` to Poisson-subsample the selected
      participations; this is folded into each LLR as ``l -> log(1 - q + q
      e^l)`` with the K-subset recurrence left unchanged.
    sample: A single observation of shape ``(iterations,)`` (``mode + noise``
      for a positive sample, pure noise for a negative one). Batch with
      ``jax.vmap`` over the sample axis.
    noise_multiplier: Standard deviation ``sigma`` of the Gaussian noise.
    c_col: The nonzero first-column band of the Toeplitz matrix ``C``, with
      ``len(c_col) <= b`` so participation bands do not overlap.

  Returns:
    The scalar privacy loss ``log[p(present) / p(absent)]``. For a negative
    sample the caller should negate the result.

  Raises:
    ValueError: For unsupported strategies or a non-1D ``sample``.
  """
  sample = jnp.asarray(sample)
  c_col = jnp.asarray(c_col, dtype=sample.dtype)
  if sample.ndim != 1:
    raise ValueError('sample must be 1D of shape (iterations,).')
  if c_col.shape[0] > sample.shape[0]:
    c_col = c_col[: sample.shape[0]]
  llrs = _log_likelihood_ratios(c_col, sample, noise_multiplier)

  if isinstance(strategy, batch_selection.BalancedMinSep):
    if strategy.sampling_prob < 1.0:
      # Keeping each participation w.p. q is exact via the per-position fold
      # l -> log(1 - q + q e^l); the K-subset recurrence is left unchanged.
      q = strategy.sampling_prob
      llrs = jnp.logaddexp(jnp.log1p(-q), jnp.log(q) + llrs)
    if strategy.random_rotation:
      return _balanced_min_sep_cyclic_log_loss(
          llrs, strategy.num_participations, strategy.min_sep
      )
    return _balanced_min_sep_log_loss(
        llrs, strategy.num_participations, strategy.min_sep
    )
  if isinstance(strategy, batch_selection.NestedRandomAllocation):
    if strategy.sampling_prob < 1.0:
      # Keeping each participation w.p. q is exact via the per-position fold
      # l -> log(1 - q + q e^l); the K-subset recurrence is left unchanged.
      q = strategy.sampling_prob
      llrs = jnp.logaddexp(jnp.log1p(-q), jnp.log(q) + llrs)
    return _nested_random_allocation_log_loss(
        llrs, strategy.num_participations, strategy.cycle_length
    )
  raise ValueError(f'Unsupported strategy: {type(strategy).__name__}')
