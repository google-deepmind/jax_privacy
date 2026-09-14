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

Per included privacy unit:
  * compute the per-sample gradient (via `jax_privacy.clipped_grad`)
  * restrict to a caller-provided set of candidate pytree leaves
  * take L2 norm per candidate leaf
  * vote +1 on the top-`vote_top_k` leaves

The vote vectors are summed across the caller-provided probe batch and Gaussian
noise is added directly to the histogram; the top `select_top_k` layers by
noisy vote count are the selected set. The probe runs once, separately from the
downstream jitted training step. Only its batched gradient computation is
jitted.

This low-level helper deliberately does not sample records or construct a
`dp_accounting.DpEvent`. The caller must use a sampling mechanism appropriate
for its privacy definition and account for that mechanism separately. For
example, one independent Poisson draw with probability `q` can be represented
by `accounting.dpsgd_event(noise_multiplier, 1, sampling_prob=q)`.

DP analysis: each included privacy unit contributes a vote vector in
`{0,1}^L` with exactly `vote_top_k` ones, so the L2 sensitivity under
ADD_OR_REMOVE is `sqrt(vote_top_k)`. Each unit must be independently sampled
if the caller claims Poisson amplification. The Gaussian noise added has stddev
`noise_multiplier * sqrt(vote_top_k)`, matching the standard convention that
`noise_multiplier` is expressed in units of sensitivity.

Example usage (not runnable as a doctest — caller supplies `params`,
`loss_fn`, `full_dataset`, `train_size`, `sampling_rng`, and `noise_key` from
their model + dataset)::

    import dp_accounting
    import jax
    import jax_privacy
    import optax
    from jax_privacy import batch_selection
    from jax_privacy import saliency

    # Boolean pytree same shape as params; True on candidate leaves.
    candidate_mask = jax.tree.map(lambda p: p.ndim == 2, params)

    sampling = batch_selection.CyclicPoissonSampling(
        sampling_prob=1024 / train_size,
        iterations=1,
        partition_type=batch_selection.PartitionType.INDEPENDENT,
    )
    indices = next(sampling.batch_iterator(train_size, rng=sampling_rng))
    probe_batch = jax.tree.map(lambda x: x[indices], full_dataset)

    result = saliency.topk_vote_probe(
        loss_fn=loss_fn,
        dataset=probe_batch,
        params=params,
        vote_top_k=8,
        select_top_k=16,
        noise_multiplier=6.0,
        candidate_mask=candidate_mask,
        prng_key=noise_key,
    )

    # Sampling and accounting are caller-owned. This event matches the
    # independent Poisson draw above and the probe's noise multiplier.
    probe_event = jax_privacy.accounting.dpsgd_event(
        noise_multiplier=6.0,
        iterations=1,
        sampling_prob=sampling.sampling_prob,
    )

    freeze_mask = jax.tree.map(lambda selected: not selected,
                               result.selected_mask)
    optimizer = optax.selective_transform(
        optax.adam(1e-3), freeze_mask=freeze_mask
    )
    total_event = dp_accounting.ComposedDpEvent(
        [probe_event, jax_privacy.accounting.dpsgd_event(...)]
    )
"""

import dataclasses
import functools
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import clipping


@dataclasses.dataclass(frozen=True)
class ProbeResult:
  """Return value of `topk_vote_probe`.

  Attributes:
    selected_mask: Boolean pytree with the same structure as the caller's
      `candidate_mask`. `True` on leaves selected as top by noisy vote count;
      `False` elsewhere (including on non-candidate leaves).
    ranked_scores: Descending-sorted list of `(candidate_index, noisy_score)`
      tuples. `candidate_index` is the position of the leaf among candidates
      (in the canonical flattening order of `candidate_mask`).
  """

  selected_mask: Any
  ranked_scores: list[tuple[int, float]]


def _mask_from_selected(
    candidate_mask: Any, selected_candidate_indices: set[int]
) -> Any:
  """Builds a mask from indices in the flattened sequence of candidates."""
  flat, treedef = jax.tree.flatten(candidate_mask)
  new_flat = []
  candidate_index = 0
  for is_candidate in flat:
    if bool(is_candidate):
      new_flat.append(candidate_index in selected_candidate_indices)
      candidate_index += 1
    else:
      new_flat.append(False)
  return jax.tree.unflatten(treedef, new_flat)


def _vote_transform(
    grads_pytree: Any,
    *,
    candidate_mask: Any,
    vote_top_k: int,
) -> jax.Array:
  """Converts one example's gradient PyTree to a candidate top-k vote."""
  candidate_grads = [
      grad
      for grad, is_candidate in zip(
          jax.tree.leaves(grads_pytree),
          jax.tree.leaves(candidate_mask),
          strict=True,
      )
      if bool(is_candidate)
  ]
  norms = jnp.stack(
      [jnp.linalg.norm(grad.astype(jnp.float32)) for grad in candidate_grads]
  )
  _, top_idx = jax.lax.top_k(norms, k=vote_top_k)
  return jax.nn.one_hot(top_idx, len(candidate_grads), dtype=jnp.float32).sum(0)


def _pad_batch_for_microbatching(
    dataset: optax.ArrayTree,
    microbatch_size: int | None,
) -> tuple[optax.ArrayTree, jax.Array]:
  """Pads a non-empty batch and returns its padding indicator."""
  batch_size = _validate.batch(dataset)
  indices = np.arange(batch_size)
  if microbatch_size is not None:
    indices = batch_selection.pad_to_multiple_of(
        indices,
        multiple=microbatch_size,
        microbatch_size=microbatch_size,
    )
  is_padding_example = indices < 0
  safe_indices = np.maximum(indices, 0)
  dataset = jax.tree.map(lambda x: x[safe_indices], dataset)
  return dataset, jnp.asarray(is_padding_example)


def topk_vote_probe(
    loss_fn: Callable[..., jax.Array],
    dataset: optax.ArrayTree,
    params: optax.ArrayTree,
    *,
    vote_top_k: int,
    select_top_k: int,
    noise_multiplier: float,
    candidate_mask: Any,
    prng_key: jax.Array,
    microbatch_size: int | None = 1,
) -> ProbeResult:
  """Runs the DP top-k voting probe and returns a selection mask.

  This is a one-time pre-training mechanism, separate from the downstream
  jitted training step. The function computes per-sample gradients via
  `jax_privacy.clipped_grad`, extracts a one-hot top-`vote_top_k` vote vector
  per example over the leaves selected by `candidate_mask`, sums the vote
  vectors, and adds Gaussian noise. The expensive clipped-gradient call is
  jitted, and its built-in microbatching performs sequential accumulation.

  Sampling and privacy accounting are intentionally outside this low-level
  API. The caller must select the records in `dataset` and construct a
  `DpEvent` matching the actual sampling mechanism and `noise_multiplier`. For
  independent Poisson sampling under add-or-remove adjacency, use one
  Poisson-sampled Gaussian event. Without valid sampling amplification, account
  for the unsampled Gaussian mechanism instead. Sampling randomness, sampled
  indices, the realized Poisson batch size, and `prng_key` must not be released
  or predictable.

  Args:
    loss_fn: The per-example loss. `loss_fn(params, *batch_args) -> loss`
      following the same convention as `jax_privacy.clipped_grad`.
    dataset: A single caller-selected batched pytree. Every leaf must have the
      same leading dimension.
    params: The model parameters (a pytree). Only used to compute gradients;
      not updated.
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
    prng_key: Private, unpredictable PRNG key for the Gaussian noise.
    microbatch_size: The vmap width used internally by `clipped_grad`. The
      input batch is padded automatically when necessary. Trades peak memory
      (larger) for wall-clock time (smaller is slower).

  Returns:
    A `ProbeResult`.

  Raises:
    ValueError: If an argument, batch, or mask is invalid.
  """
  _validate.tree_structure(params, candidate_mask=candidate_mask)
  batch_size = _validate.batch(dataset)
  num_candidates = sum(bool(x) for x in jax.tree.leaves(candidate_mask))
  _validate.positive(num_candidates=num_candidates)
  _validate.in_range(
      1,
      num_candidates,
      vote_top_k=vote_top_k,
      select_top_k=select_top_k,
  )
  _validate.non_negative(noise_multiplier=noise_multiplier)
  if microbatch_size is not None:
    _validate.positive(microbatch_size=microbatch_size)

  l2_sensitivity = math.sqrt(vote_top_k)
  vote_transform = functools.partial(
      _vote_transform,
      candidate_mask=candidate_mask,
      vote_top_k=vote_top_k,
  )

  # clipped_grad with L2 clip == sensitivity is a no-op for vote vectors
  # (they are exactly on the L2 ball of radius sqrt(vote_top_k) by
  # construction), but it lets us reuse the standard per-example grad + sum
  # pipeline without extra machinery.
  grad_fn = clipping.clipped_grad(
      loss_fn,
      argnums=0,
      batch_argnums=1,
      l2_clip_norm=l2_sensitivity,
      pre_clipping_transform=vote_transform,
      microbatch_size=microbatch_size,
  )

  if batch_size:
    dataset, is_padding_example = _pad_batch_for_microbatching(
        dataset, microbatch_size
    )
    total_votes = jax.jit(grad_fn)(
        params, dataset, is_padding_example=is_padding_example
    )
  else:
    total_votes = jnp.zeros((num_candidates,), dtype=jnp.float32)

  sensitivity = grad_fn.sensitivity()
  noise = jax.random.normal(
      prng_key, total_votes.shape, dtype=total_votes.dtype
  )
  noisy_votes = total_votes + noise_multiplier * sensitivity * noise

  # Rank candidates by noisy vote count.
  scores = jax.device_get(noisy_votes).tolist()
  ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
  selected_local_indices = {i for i, _ in ranked[:select_top_k]}
  selected_mask = _mask_from_selected(candidate_mask, selected_local_indices)

  return ProbeResult(
      selected_mask=selected_mask,
      ranked_scores=ranked,
  )
