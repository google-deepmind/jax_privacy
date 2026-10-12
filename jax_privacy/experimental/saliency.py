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

Implements the discrete top-k voting probe of DP-SAPF, see
`Gong et al. (2026) <https://arxiv.org/abs/2605.30312>`_. The probe privately
selects a small subset of parameter leaves with the largest per-example
gradient norms, so that downstream DP-SGD only needs to train (and add noise
to) that subset.

Per included privacy unit:

* compute the per-example gradient via
  :func:`~jax_privacy.clipping.clipped_grad`,
* restrict to the candidate leaves selected by ``candidate_mask``,
* take the L2 norm of each candidate leaf,
* vote +1 on the top ``vote_top_k`` leaves.

The vote vectors are summed across the probe batch and Gaussian noise is added
to the resulting histogram; the top ``select_top_k`` leaves by noisy vote count
form the selected set. The probe runs once, before training; only its batched
gradient computation is jitted.

DP analysis: each privacy unit contributes a vote vector in ``{0, 1}^L`` with
exactly ``vote_top_k`` ones, so the L2 sensitivity under add-or-remove
adjacency is ``sqrt(vote_top_k)``. The added Gaussian noise has stddev
``noise_multiplier * sqrt(vote_top_k)``, following the standard convention
that ``noise_multiplier`` is expressed in units of sensitivity.

This low-level helper deliberately does not sample records or construct a
``dp_accounting.DpEvent``. The caller selects the probe batch with a sampling
mechanism appropriate for its privacy definition and accounts for it
separately; see :func:`topk_vote_probe` for an example.
"""

from collections.abc import Mapping
import dataclasses
import functools
import itertools
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import clipping
import numpy as np
import optax


@dataclasses.dataclass(frozen=True)
class ProbeResult:
  """Return value of :func:`topk_vote_probe`.

  Attributes:
    selected_mask: Boolean pytree with the same structure as the caller's
      ``candidate_mask``. ``True`` on the ``select_top_k`` leaves with the
      highest noisy vote count; ``False`` elsewhere (including on non-candidate
      leaves).
    ranked_scores: ``(candidate_index, noisy_score)`` tuples sorted by
      descending score. ``candidate_index`` is the position of the leaf among
      the candidates, in the canonical flattening order of ``candidate_mask``.
  """

  selected_mask: Any
  ranked_scores: list[tuple[int, float]]


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
  if microbatch_size is None:
    return dataset, jnp.zeros((batch_size,), dtype=jnp.bool_)
  indices = batch_selection.pad_to_multiple_of(
      np.arange(batch_size),
      multiple=microbatch_size,
      microbatch_size=microbatch_size,
  )
  is_padding_example = indices < 0
  safe_indices = np.maximum(indices, 0)
  dataset = jax.tree.map(lambda x: x[safe_indices], dataset)
  return dataset, jnp.asarray(is_padding_example)


def topk_vote_probe(
    loss_fn: Callable[..., Any],
    dataset: optax.ArrayTree,
    params: optax.ArrayTree,
    *,
    vote_top_k: int,
    select_top_k: int,
    noise_multiplier: float,
    candidate_mask: Any,
    prng_key: jax.Array,
    microbatch_size: int | None = None,
    clipping_kwargs: Mapping[str, Any] | None = None,
) -> ProbeResult:
  """Runs the DP top-k voting probe and returns a selection mask.

  See the module docstring for the mechanism and its sensitivity analysis.
  Sampling and privacy accounting are intentionally left to the caller: pass
  an already-sampled ``dataset`` and construct the matching ``DpEvent``
  yourself. Sampling randomness, the sampled indices, the realized batch size
  and ``prng_key`` must remain private.

  Example:
    Noise is disabled here only to keep the output deterministic.

    >>> import dp_accounting
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jax_privacy import accounting
    >>> from jax_privacy.experimental import saliency
    >>> params = (jnp.array(0.0), jnp.array(0.0))
    >>> data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> def loss_fn(params, batch):
    ...   return jnp.mean(batch @ jnp.stack(params))
    >>> result = saliency.topk_vote_probe(
    ...     loss_fn, data, params,
    ...     vote_top_k=1, select_top_k=1, noise_multiplier=0.0,
    ...     candidate_mask=(True, True), prng_key=jax.random.key(0),
    ... )
    >>> result.selected_mask
    (False, True)
    >>> result.ranked_scores
    [(1, 2.0), (0, 0.0)]

    If ``data`` was one Poisson draw with probability ``q`` and the probe used
    ``noise_multiplier=sigma``, its event composes with downstream DP-SGD
    like any other sampled Gaussian mechanism:

    >>> q, sigma = 0.1, 1.0
    >>> probe_event = accounting.dpsgd_event(sigma, 1, sampling_prob=q)
    >>> train_event = accounting.dpsgd_event(sigma, 100, sampling_prob=q)
    >>> accountant = dp_accounting.pld.PLDAccountant()
    >>> _ = accountant.compose(probe_event).compose(train_event)
    >>> accountant.get_epsilon(target_delta=1e-6) > 0
    True

  Args:
    loss_fn: The loss, called as ``loss_fn(params, batch)`` by default. It must
      return a scalar, or ``(loss, aux)`` when ``clipping_kwargs`` sets
      ``has_aux=True``; auxiliary outputs are discarded. Bind any other
      arguments in this callable.
    dataset: A single caller-selected batched pytree. Every leaf must have the
      same leading dimension.
    params: The model parameters (a pytree). Only used to compute gradients.
    vote_top_k: Number of candidate leaves each example votes for. Determines
      the L2 sensitivity ``sqrt(vote_top_k)``.
    select_top_k: Number of candidate leaves to keep after ranking by noisy vote
      count. Must be at most the number of candidates.
    noise_multiplier: Gaussian noise stddev in units of the L2 sensitivity.
    candidate_mask: Boolean pytree with the same structure as ``params``. Leaves
      set to ``True`` are candidates for selection; all other leaves receive no
      votes and are ``False`` in the returned ``selected_mask``.
    prng_key: Private, unpredictable PRNG key for the Gaussian noise.
    microbatch_size: The vmap width used internally by
      :func:`~jax_privacy.clipping.clipped_grad`. ``None`` vectorizes over the
      entire batch; a positive integer bounds peak memory by accumulating
      microbatches sequentially (the batch is padded automatically).
    clipping_kwargs: Optional keywords forwarded to
      :func:`~jax_privacy.clipping.clipped_grad`, e.g. ``has_aux``,
      ``keep_batch_dim=False`` for a per-example loss without a batch axis, or
      ``argnums=1, batch_argnums=0`` for ``loss_fn(batch, params)``. The probe
      reserves ``l2_clip_norm``, ``pre_clipping_transform``,
      ``microbatch_size``, ``normalize_by``, ``rescale_to_unit_norm``,
      ``grid_scale`` and ``slack``.

  Returns:
    A :class:`ProbeResult`.

  Raises:
    ValueError: If an argument, batch, or mask is invalid, or
      ``clipping_kwargs`` contains a reserved option or unsupported argument
      positions.
  """
  clipping_kwargs = dict(clipping_kwargs or {})
  reserved = clipping_kwargs.keys() & {
      'l2_clip_norm',
      'pre_clipping_transform',
      'microbatch_size',
      'normalize_by',
      'rescale_to_unit_norm',
      'grid_scale',
      'slack',
  }
  if reserved:
    raise ValueError(
        f'The probe controls these clipping_kwargs: {sorted(reserved)}.'
    )
  argnums = clipping_kwargs.get('argnums', 0)
  batch_argnums = clipping_kwargs.get('batch_argnums', 1)
  if (
      not isinstance(argnums, int)
      or not isinstance(batch_argnums, int)
      or (argnums, batch_argnums) not in ((0, 1), (1, 0))
  ):
    raise ValueError(
        'clipping_kwargs argnums and batch_argnums must be distinct integer '
        'positions 0 and 1 for the parameters and dataset.'
    )
  _validate.tree_structure(params, candidate_mask=candidate_mask)
  batch_size = _validate.batch(dataset)
  num_candidates = sum(bool(x) for x in jax.tree.leaves(candidate_mask))
  _validate.positive(num_candidates=num_candidates)
  _validate.is_int(vote_top_k=vote_top_k, select_top_k=select_top_k)
  _validate.in_range(
      1,
      num_candidates,
      vote_top_k=vote_top_k,
      select_top_k=select_top_k,
  )
  _validate.non_negative(noise_multiplier=noise_multiplier)
  if microbatch_size is not None:
    _validate.positive(microbatch_size=microbatch_size)

  vote_transform = functools.partial(
      _vote_transform,
      candidate_mask=candidate_mask,
      vote_top_k=vote_top_k,
  )
  # Vote vectors lie exactly on the L2 ball of radius sqrt(vote_top_k), so
  # clipping is a no-op; clipped_grad is used for its per-example grad + sum
  # pipeline (vmap, microbatching, padding, sensitivity bookkeeping).
  grad_fn = clipping.clipped_grad(
      loss_fn,
      l2_clip_norm=math.sqrt(vote_top_k),
      pre_clipping_transform=vote_transform,
      microbatch_size=microbatch_size,
      **clipping_kwargs,
  )

  def aggregate_votes(params, dataset, is_padding_example):
    args = (params, dataset) if argnums == 0 else (dataset, params)
    output = grad_fn(*args, is_padding_example=is_padding_example)
    return output[0] if grad_fn.has_aux else output

  if batch_size:
    dataset, is_padding_example = _pad_batch_for_microbatching(
        dataset, microbatch_size
    )
    total_votes = jax.jit(aggregate_votes)(
        params, dataset, is_padding_example=is_padding_example
    )
  else:
    total_votes = jnp.zeros((num_candidates,), dtype=jnp.float32)

  noise = jax.random.normal(
      prng_key, total_votes.shape, dtype=total_votes.dtype
  )
  noisy_votes = total_votes + noise_multiplier * grad_fn.sensitivity() * noise

  scores = jax.device_get(noisy_votes).tolist()
  ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
  selected = {i for i, _ in ranked[:select_top_k]}
  # jax.tree.map visits leaves in the same order as jax.tree.leaves, so the
  # counter assigns each candidate leaf its index among the candidates.
  counter = itertools.count()
  selected_mask = jax.tree.map(
      lambda is_candidate: bool(is_candidate) and next(counter) in selected,
      candidate_mask,
  )
  return ProbeResult(selected_mask=selected_mask, ranked_scores=ranked)
