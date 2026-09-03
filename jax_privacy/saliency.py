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

"""DP saliency probe for parameter-selective DP fine-tuning.

Implements the discrete top-k voting probe of DP-SAPF (Gong, Li, Lin, Wang,
2026, "DP-SAPF: Saliency-Aware Parameter Fine-tuning of Public Models for
Differentially Private Image Synthesis", USENIX Security 2026,
https://arxiv.org/abs/2605.30312).

This helper is an LLM-oriented top-k voting adaptation of DP-SAPF rather than
a verbatim implementation of the paper's image-synthesis method. Prefer the
probe when a public pretrained model has many candidate parameter leaves and
saliency is expected to be concentrated in a small, stable subset, so the
one-time probe cost can be offset by a smaller downstream trainable set. A
standard all-candidate or default-LoRA baseline may be preferable when the
candidate set is already small, saliency is diffuse or unstable, or the
additional privacy and computation cost of the probe cannot be amortized.
Utility comparisons should use the same total privacy budget.

Per training sample:
  * compute the per-sample gradient (via `jax_privacy.clipped_grad`)
  * restrict to a caller-provided set of candidate pytree leaves
  * take L2 norm per candidate leaf
  * vote +1 on the top-`vote_top_k` leaves

The vote vectors are summed across the probe dataset and Gaussian noise is
added to the histogram (via `noise_addition.gaussian_privatizer`); the top
`select_top_k` layers by noisy vote count are the selected set. The returned
mask is a boolean pytree in the same structure as `candidate_mask`, directly
usable with `optax.masked` for downstream DP-SGD fine-tuning.

DP analysis: each user contributes a vote vector in `{0,1}^L` with exactly
`vote_top_k` ones, so the L2 sensitivity under ADD_OR_REMOVE is
`sqrt(vote_top_k)`. The Gaussian noise added has stddev
`noise_multiplier * sqrt(vote_top_k)`, matching the standard convention that
`noise_multiplier` is expressed in units of sensitivity.

Example usage (not runnable as a doctest — caller supplies `params`,
`loss_fn`, `probe_batches`, `train_size` from their own model + dataset)::

    import jax
    import jax_privacy
    import optax
    from jax_privacy import saliency

    # Boolean pytree same shape as params; True on candidate leaves.
    candidate_mask = jax.tree.map(lambda p: p.ndim == 2, params)

    result = saliency.topk_vote_probe(
        loss_fn=loss_fn,
        dataset=probe_batches,          # iterable of microbatch-sized batches
        params=params,
        num_samples=1024,
        vote_top_k=8,
        select_top_k=16,
        noise_multiplier=6.0,
        candidate_mask=candidate_mask,
        prng_key=jax.random.PRNGKey(0),
        sampling_probability=1024 / train_size,
    )

    masked_optimizer = optax.masked(optax.adam(1e-3), result.selected_mask)
    # then compose accounting: dp_accounting.ComposedDpEvent(
    #    [result.dp_event, jax_privacy.accounting.dpsgd_event(...)])
"""

import dataclasses
import math
from typing import Any, Callable, Iterable

import dp_accounting
import jax
import jax.numpy as jnp

from jax_privacy import clipping


@dataclasses.dataclass(frozen=True)
class ProbeResult:
  """Return value of `topk_vote_probe`.

  Attributes:
    selected_mask: Boolean pytree with the same structure as the caller's
      `candidate_mask`. `True` on leaves selected as top by noisy vote count;
      `False` elsewhere (including on non-candidate leaves). Directly usable
      as the `mask` argument of `optax.masked`.
    ranked_scores: Descending-sorted list of `(candidate_index, noisy_score)`
      tuples. `candidate_index` is the position of the leaf among candidates
      (in the canonical flattening order of `candidate_mask`).
    n_seen: Number of probe samples actually processed (may be less than
      `num_samples` if the dataset was exhausted early).
    dp_event: The `dp_accounting.DpEvent` representing the probe's privacy
      cost. Compose with the DP-SGD training event via
      `dp_accounting.ComposedDpEvent([dp_event, training_event])`.
  """

  selected_mask: Any
  ranked_scores: list[tuple[int, float]]
  n_seen: int
  dp_event: dp_accounting.DpEvent


def _candidate_positions(candidate_mask: Any) -> tuple[int, ...]:
  """Positions of True entries in the flattened `candidate_mask`."""
  flat, _ = jax.tree_util.tree_flatten(candidate_mask)
  # bool leaves in the mask pytree may be Python bools, JAX arrays, or numpy;
  # coerce to Python bool for a static tuple usable at trace time.
  return tuple(i for i, m in enumerate(flat) if bool(m))


def _mask_from_selected(
    candidate_mask: Any, selected_positions: set[int]
) -> Any:
  """Builds a same-structure boolean pytree, True only on selected leaves."""
  flat, treedef = jax.tree_util.tree_flatten(candidate_mask)
  new_flat = [i in selected_positions for i in range(len(flat))]
  return jax.tree_util.tree_unflatten(treedef, new_flat)


def _make_vote_transform(
    candidate_positions: tuple[int, ...], vote_top_k: int
) -> Callable[[Any], jax.Array]:
  """Returns a pre_clipping_transform: per-example grad -> top-k vote vector."""
  num_candidates = len(candidate_positions)

  def _vote_transform(grads_pytree: Any) -> jax.Array:
    flat = jax.tree_util.tree_leaves(grads_pytree)
    norms = jnp.stack([
        jnp.linalg.norm(flat[i].astype(jnp.float32))
        for i in candidate_positions
    ])
    _, top_idx = jax.lax.top_k(norms, k=vote_top_k)
    return jax.nn.one_hot(top_idx, num_candidates, dtype=jnp.float32).sum(0)

  return _vote_transform


def topk_vote_probe(
    loss_fn: Callable[..., jax.Array],
    dataset: Iterable[Any],
    params: Any,
    *,
    num_samples: int,
    vote_top_k: int,
    select_top_k: int,
    noise_multiplier: float,
    candidate_mask: Any,
    prng_key: jax.Array,
    sampling_probability: float,
    microbatch_size: int = 1,
    use_noise: bool = True,
    batch_argnums: int | tuple[int, ...] = 1,
) -> ProbeResult:
  """Runs the DP top-k voting probe and returns a selection mask.

  Streams microbatches of size `microbatch_size` from `dataset` (up to
  `num_samples` examples), computes per-sample gradients via
  `jax_privacy.clipped_grad`, extracts a one-hot top-`vote_top_k` vote vector
  per example over the leaves selected by `candidate_mask`, sums the vote
  vectors, and adds Gaussian noise with stddev `noise_multiplier *
  sqrt(vote_top_k)`. Returns a boolean pytree `selected_mask` with `True` on
  the `select_top_k` candidate leaves whose noisy vote count is largest.

  Args:
    loss_fn: The per-example loss. `loss_fn(params, *batch_args) -> loss`
      following the same convention as `jax_privacy.clipped_grad`.
    dataset: An iterable yielding microbatches, each already batched along its
      leading axis to size `microbatch_size`. Iteration stops once
      `num_samples` examples have been consumed or the iterator is exhausted.
    params: The model parameters (a pytree). Only used to compute gradients;
      not updated.
    num_samples: The total number of probe samples to consume.
    vote_top_k: Per-sample voting width. Each sample votes +1 on this many
      candidate leaves. Determines the L2 sensitivity of the mechanism
      (`sqrt(vote_top_k)`).
    select_top_k: Number of candidate leaves to keep after ranking by noisy
      vote count. Must be `<= sum(candidate_mask)`.
    noise_multiplier: Gaussian noise stddev in units of the L2 sensitivity.
      The absolute per-bin stddev of the added noise is
      `noise_multiplier * sqrt(vote_top_k)`.
    candidate_mask: Boolean pytree with the same structure as `params`. Leaves
      set to `True` are considered candidates for selection. Non-candidate
      leaves get no votes and are `False` in the returned `selected_mask`.
    prng_key: Base PRNG key for the Gaussian noise.
    sampling_probability: The Poisson-subsampling probability that produced
      the probe dataset (usually `num_samples / train_size`). Used only to
      build the returned `DpEvent`. It is the caller's responsibility to
      ensure the actual sampling matches this value.
    microbatch_size: The vmap width used inside `clipped_grad`. Trades peak
      memory (larger) for wall-clock (smaller is slower).
    use_noise: If `False`, skips the Gaussian noise addition. The returned
      `dp_event` still describes the mechanism as if noise had been added;
      the caller loses the DP guarantee in exchange for a deterministic
      selection (useful for reproducibility experiments).
    batch_argnums: Which argument(s) of `loss_fn` carry the batch axis;
      passed through to `jax_privacy.clipped_grad` unchanged.

  Returns:
    A `ProbeResult`.

  Raises:
    ValueError: if `candidate_mask` has no `True` leaves, or if `vote_top_k`
      or `select_top_k` exceed the number of candidates, or if
      `sampling_probability` is not in `(0, 1]`.
  """
  candidate_positions = _candidate_positions(candidate_mask)
  num_candidates = len(candidate_positions)
  if num_candidates == 0:
    raise ValueError('candidate_mask has no True leaves.')
  if vote_top_k > num_candidates:
    raise ValueError(
        f'vote_top_k={vote_top_k} exceeds num_candidates={num_candidates}.'
    )
  if select_top_k > num_candidates:
    raise ValueError(
        f'select_top_k={select_top_k} exceeds num_candidates={num_candidates}.'
    )
  if not 0.0 < sampling_probability <= 1.0:
    raise ValueError(
        f'sampling_probability must be in (0, 1]; got {sampling_probability}.'
    )

  l2_sensitivity = math.sqrt(vote_top_k)
  vote_transform = _make_vote_transform(candidate_positions, vote_top_k)

  # clipped_grad with L2 clip == sensitivity is a no-op for vote vectors
  # (they are exactly on the L2 ball of radius sqrt(vote_top_k) by
  # construction), but it lets us reuse the standard per-example grad + sum
  # pipeline without extra machinery.
  grad_fn = clipping.clipped_grad(
      loss_fn,
      argnums=0,
      batch_argnums=batch_argnums,
      l2_clip_norm=l2_sensitivity,
      pre_clipping_transform=vote_transform,
      microbatch_size=microbatch_size,
  )

  # JIT the per-microbatch accumulation. Without this, each call retraces
  # the vmap + grad + pre_clipping_transform pipeline, which is catastrophic
  # for `microbatch_size=1` (one retrace per sample). Consistent batch shapes
  # (guaranteed by callers using `drop_remainder=True`) let the traced graph
  # be reused across chunks.
  @jax.jit
  def _accumulate(votes, params, batch):
    return votes + grad_fn(params, batch)

  total_votes = jnp.zeros((num_candidates,), dtype=jnp.float32)
  n_seen = 0
  data_iter = iter(dataset)
  while n_seen < num_samples:
    try:
      batch = next(data_iter)
    except StopIteration:
      break
    total_votes = _accumulate(total_votes, params, batch)
    # We assume each yielded batch is exactly `microbatch_size`; if the
    # caller yields ragged batches the accounting will be over-counted by
    # at most `microbatch_size - 1` examples.
    n_seen += microbatch_size

  if use_noise:
    # `noise_addition.gaussian_privatizer` is designed for T-step DP-SGD
    # loops (it initialises streaming-matrix state whose size scales with T)
    # and over-allocates when applied to our one-shot histogram noise. A
    # direct Gaussian draw with the correct stddev gives the identical
    # mechanism at trivial cost.
    noise = (noise_multiplier * l2_sensitivity) * jax.random.normal(
        prng_key, (num_candidates,), dtype=jnp.float32
    )
    noisy_votes = total_votes + noise
  else:
    noisy_votes = total_votes

  # Rank candidates by noisy vote count.
  scores = jax.device_get(noisy_votes).tolist()
  ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
  selected_local_indices = {i for i, _ in ranked[:select_top_k]}
  selected_flat_positions = {
      candidate_positions[i] for i in selected_local_indices
  }
  selected_mask = _mask_from_selected(candidate_mask, selected_flat_positions)

  dp_event: dp_accounting.DpEvent = dp_accounting.PoissonSampledDpEvent(
      sampling_probability=sampling_probability,
      event=dp_accounting.GaussianDpEvent(noise_multiplier),
  )

  return ProbeResult(
      selected_mask=selected_mask,
      ranked_scores=ranked,
      n_seen=n_seen,
      dp_event=dp_event,
  )
