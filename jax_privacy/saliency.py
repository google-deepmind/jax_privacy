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

Per sampled privacy unit:
  * compute the per-sample gradient (via `jax_privacy.clipped_grad`)
  * restrict to a caller-provided set of candidate pytree leaves
  * take L2 norm per candidate leaf
  * vote +1 on the top-`vote_top_k` leaves

The vote vectors are summed across a Poisson-sampled probe batch and Gaussian
noise is added to the histogram (via `noise_addition.gaussian_privatizer`); the
top `select_top_k` layers by noisy vote count are the selected set. The probe
runs once, separately from the downstream jitted training step. Only its
batched gradient computation is jitted.

DP analysis: each included privacy unit contributes a vote vector in
`{0,1}^L` with exactly `vote_top_k` ones, so the L2 sensitivity under
ADD_OR_REMOVE is `sqrt(vote_top_k)`. Each unit must be independently sampled
with the public probability in `sampling_strategy`. The Gaussian noise added
has stddev `noise_multiplier * sqrt(vote_top_k)`, matching the standard
convention that `noise_multiplier` is expressed in units of sensitivity.

Example usage (not runnable as a doctest — caller supplies `params`,
`loss_fn`, `full_dataset`, and `train_size` from their model + dataset)::

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
    indices = next(sampling.batch_iterator(train_size, rng=0))
    probe_batch = jax.tree.map(lambda x: x[indices], full_dataset)

    result = saliency.topk_vote_probe(
        loss_fn=loss_fn,
        dataset=probe_batch,
        params=params,
        vote_top_k=8,
        select_top_k=16,
        noise_multiplier=6.0,
        candidate_mask=candidate_mask,
        prng_key=jax.random.PRNGKey(0),
        sampling_strategy=sampling,
    )

    freeze_mask = jax.tree.map(lambda selected: not selected,
                               result.selected_mask)
    optimizer = optax.selective_transform(
        optax.adam(1e-3), freeze_mask=freeze_mask
    )
    # then compose accounting: dp_accounting.ComposedDpEvent(
    #    [result.dp_event, jax_privacy.accounting.dpsgd_event(...)])
"""

import dataclasses
import math
from typing import Any, Callable

import dp_accounting
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import clipping
from jax_privacy import noise_addition


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
    dp_event: The `dp_accounting.DpEvent` representing the probe's privacy
      cost. Compose with the DP-SGD training event via
      `dp_accounting.ComposedDpEvent([dp_event, training_event])`.
  """

  selected_mask: Any
  ranked_scores: list[tuple[int, float]]
  dp_event: dp_accounting.DpEvent


def _candidate_positions(candidate_mask: Any) -> tuple[int, ...]:
  """Positions of True entries in the flattened `candidate_mask`."""
  flat, _ = jax.tree.flatten(candidate_mask)
  # bool leaves in the mask pytree may be Python bools, JAX arrays, or numpy;
  # coerce to Python bool for a static tuple usable at trace time.
  return tuple(i for i, m in enumerate(flat) if bool(m))


def _mask_from_selected(
    candidate_mask: Any, selected_positions: set[int]
) -> Any:
  """Builds a same-structure boolean pytree, True only on selected leaves."""
  flat, treedef = jax.tree.flatten(candidate_mask)
  new_flat = [i in selected_positions for i in range(len(flat))]
  return jax.tree.unflatten(treedef, new_flat)


def _make_vote_transform(
    candidate_positions: tuple[int, ...], vote_top_k: int
) -> Callable[[Any], jax.Array]:
  """Returns a pre_clipping_transform: per-example grad -> top-k vote vector."""
  num_candidates = len(candidate_positions)

  def _vote_transform(grads_pytree: Any) -> jax.Array:
    flat = jax.tree.leaves(grads_pytree)
    norms = jnp.stack([
        jnp.linalg.norm(flat[i].astype(jnp.float32))
        for i in candidate_positions
    ])
    _, top_idx = jax.lax.top_k(norms, k=vote_top_k)
    return jax.nn.one_hot(top_idx, num_candidates, dtype=jnp.float32).sum(0)

  return _vote_transform


def _validate_sampling_strategy(
    strategy: batch_selection.CyclicPoissonSampling,
) -> None:
  """Validates that strategy represents one standard Poisson sample."""
  _validate.instance(
      batch_selection.CyclicPoissonSampling, sampling_strategy=strategy
  )
  _validate.positive(sampling_probability=strategy.sampling_prob)
  _validate.equal(
      1,
      sampling_iterations=strategy.iterations,
      sampling_cycle_length=strategy.cycle_length,
  )
  _validate.equal(None, truncated_batch_size=strategy.truncated_batch_size)
  _validate.equal(
      batch_selection.PartitionType.INDEPENDENT,
      sampling_partition_type=strategy.partition_type,
  )


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
    sampling_strategy: batch_selection.CyclicPoissonSampling,
    microbatch_size: int | None = 1,
) -> ProbeResult:
  """Runs the DP top-k voting probe and returns a selection mask.

  This is a one-time pre-training mechanism, separate from the downstream
  jitted training step. `dataset` must contain the result of exactly one draw
  from `sampling_strategy`. The function computes per-sample gradients via
  `jax_privacy.clipped_grad`, extracts a one-hot top-`vote_top_k` vote vector
  per example over the leaves selected by `candidate_mask`, sums the vote
  vectors, and adds Gaussian noise. The expensive clipped-gradient call is
  jitted, and its built-in microbatching performs sequential accumulation.

  The returned privacy event uses the probability from the same
  `sampling_strategy` object used to construct `dataset`. The sampling RNG,
  sampled indices, realized Poisson batch size, and noise PRNG key must not be
  released or predictable.

  Args:
    loss_fn: The per-example loss. `loss_fn(params, *batch_args) -> loss`
      following the same convention as `jax_privacy.clipped_grad`.
    dataset: A single batched pytree produced by exactly one draw from
      `sampling_strategy`. Every leaf must have the same leading dimension.
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
    sampling_strategy: The standard one-shot Poisson strategy used to produce
      `dataset`. It must have one iteration, cycle length one, no truncation,
      and `PartitionType.INDEPENDENT`.
    microbatch_size: The vmap width used internally by `clipped_grad`. The
      realized Poisson batch is padded automatically when necessary. Trades
      peak memory (larger) for wall-clock time (smaller is slower).

  Returns:
    A `ProbeResult`.

  Raises:
    ValueError: If an argument, batch, mask, or sampling strategy is invalid.
  """
  _validate.tree_structure(params, candidate_mask=candidate_mask)
  batch_size = _validate.batch(dataset)
  candidate_positions = _candidate_positions(candidate_mask)
  num_candidates = len(candidate_positions)
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
  _validate_sampling_strategy(sampling_strategy)

  l2_sensitivity = math.sqrt(vote_top_k)
  vote_transform = _make_vote_transform(candidate_positions, vote_top_k)

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

  neighboring_relation = dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
  sensitivity = grad_fn.sensitivity(neighboring_relation)
  privatizer = noise_addition.gaussian_privatizer(
      stddev=noise_multiplier * sensitivity,
      prng_key=prng_key,
  )
  noisy_votes, _ = privatizer.update(total_votes, privatizer.init(total_votes))

  # Rank candidates by noisy vote count.
  scores = jax.device_get(noisy_votes).tolist()
  ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
  selected_local_indices = {i for i, _ in ranked[:select_top_k]}
  selected_flat_positions = {
      candidate_positions[i] for i in selected_local_indices
  }
  selected_mask = _mask_from_selected(candidate_mask, selected_flat_positions)

  dp_event: dp_accounting.DpEvent = dp_accounting.PoissonSampledDpEvent(
      sampling_probability=sampling_strategy.sampling_prob,
      event=dp_accounting.GaussianDpEvent(noise_multiplier),
  )

  return ProbeResult(
      selected_mask=selected_mask,
      ranked_scores=ranked,
      dp_event=dp_event,
  )
