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

"""End-to-end training loop for differentially private training.

This module provides a general-purpose DP training loop driven by a
`DPExecutionPlan`, supporting arbitrary mechanisms.
"""

from collections.abc import Callable
from typing import Protocol, TypeAlias

import chex
import jax
import jax_privacy
from jax_privacy import batch_selection
from jax_privacy.experimental import execution_plan
from jax_privacy.experimental import optimizers as aug_optimizers
import numpy as np
import optax

Loss: TypeAlias = jax.Array
Aux: TypeAlias = chex.ArrayTree
PerExampleAux: TypeAlias = jax_privacy.clipping.AuxiliaryOutput
Batch: TypeAlias = chex.ArrayTree
Params: TypeAlias = chex.ArrayTree
OptState: TypeAlias = chex.ArrayTree
NoiseState: TypeAlias = chex.ArrayTree


class LossFn(Protocol):
  """Expected contract for loss functions used in DP training.

  Loss functions must accept ``params`` and a ``data`` batch, and return
  ``(loss, aux)``.  They may optionally accept a PRNG key as a third
  positional argument for stochastic operations (e.g., dropout).

  Any additional context the loss function needs — frozen parameters,
  model configuration, label smoothing constants, etc. — should be closed
  over before passing the function to :func:`train`::

      frozen = model.freeze(some_params)
      def my_loss(params, data, prng):
          all_params = {**frozen, **params}
          logits = model.apply(all_params, data['x'], rngs={'dropout': prng})
          return cross_entropy(logits, data['y']), {'logits': logits}

      training.train(..., loss_fn=my_loss, ...)

  **Mutable state that persists across steps is intentionally unsupported
  by this signature.**  Patterns like batch-norm running statistics or
  online accumulators that carry state from one step to the next are
  generally incompatible with differential privacy unless extreme care is
  taken, and are therefore excluded by design.  If you need such patterns,
  fold the state into ``params`` and manage it explicitly.

  Example signature::

      def loss_fn(params, data, prng):
          ...
          return loss, aux
  """

  def __call__(
      self,
      params: Params,
      data: Batch,
      prng: jax.Array,
  ) -> tuple[Loss, Aux]:
    ...


@chex.dataclass(frozen=True, kw_only=True)
class TrainingState:
  """Container for the state of the training loop."""

  step: int
  params: Params
  opt_state: OptState
  noise_state: NoiseState


def _get_num_examples(dataset: Batch) -> int:
  """Infers the total number of examples in the dataset."""
  leaves = jax.tree.leaves(dataset)
  if not leaves:
    raise ValueError('Dataset is empty or contains no leaves.')
  sizes = {leaf.shape[0] for leaf in leaves}
  if len(sizes) != 1:
    raise ValueError(
        'All dataset leaves must have the same size along axis 0, '
        f'got sizes: {sorted(sizes)}.'
    )
  return sizes.pop()


def _get_batch(dataset: Batch, indices: np.ndarray) -> tuple[Batch, np.ndarray]:
  """Retrieves a batch from a PyTree dataset, zeroing padding examples.

  Args:
    dataset: A PyTree of arrays.
    indices: A 1D array of indices. Entries equal to ``-1`` are treated as
      padding and the corresponding examples are zeroed out.

  Returns:
    A tuple ``(batch, is_padding)`` where ``batch`` is the indexed and
    zero-padded PyTree and ``is_padding`` is a boolean array indicating
    which examples are padding.
  """
  is_padding = indices == -1

  def _index_and_zero(x: np.ndarray) -> np.ndarray:
    mask = np.expand_dims(is_padding, tuple(range(1, x.ndim)))
    return np.where(mask, 0, x[indices])

  return jax.tree.map(_index_and_zero, dataset), is_padding


def train(
    plan: execution_plan.DPExecutionPlan,
    dataset: Batch,
    loss_fn: LossFn,
    params: Params,
    optimizer: (
        aug_optimizers.AugmentedGradientTransformation
        | optax.GradientTransformation
    ),
    padding_multiple: int = 1,
    callback: Callable[[int, TrainingState, PerExampleAux], None] | None = None,
    rng: np.random.Generator | int | None = None,
) -> TrainingState:
  """Runs an end-to-end differentially private training loop.

  **Sharding**: This function does not shard params or data.  For
  multi-device training, provide ``params`` with explicit sharding
  annotations and configure ``spmd_axis_name`` through the plan's
  ``PerformanceFlags``.  If data sharding is needed, ``loss_fn``
  should reshard its inputs using sharding-in-types.

  Args:
    plan: A ``DPExecutionPlan`` specifying the DP mechanism.
    dataset: The training dataset, as a PyTree of arrays.
    loss_fn: The per-example loss function.  See :class:`LossFn`.
    params: Initial parameter PyTree.
    optimizer: An ``AugmentedGradientTransformation`` or a plain
      ``optax.GradientTransformation``.
    padding_multiple: If set, batch sizes are padded to a multiple of this
      value.
    callback: Called after each step as ``callback(step, state, aux)``. ``step``
      is a Python int.
    rng: Optional random seed or ``numpy.random.Generator`` for reproducibility.

  Returns:
    Final ``TrainingState``.
  """
  optimizer = aug_optimizers.as_augmented_optimizer(optimizer)

  rng = np.random.default_rng(rng)
  loss_rng = jax.random.key(int(rng.integers(2**63)))

  num_examples = _get_num_examples(dataset)

  state = TrainingState(
      step=0,
      params=params,
      opt_state=optimizer.init(params),
      noise_state=plan.noise_addition_transform.init(params),
  )

  batch_iterator = plan.batch_selection_strategy.batch_iterator(
      num_examples, rng=rng
  )

  @jax.jit
  def step_fn(state, batch, is_padding_example):

    pre_clip_fn = optimizer.pre_clipping_transform(state.opt_state)

    grad_fn = plan.clipped_grad(
        loss_fn,
        has_aux=True,
        return_values=True,
        return_grad_norms=True,
        pre_clipping_transform=pre_clip_fn,
        prng_argnum=2,
    )

    rng = jax.random.fold_in(loss_rng, state.step)
    clipped_grad_sum, aux = grad_fn(
        state.params, batch, rng, is_padding_example=is_padding_example
    )

    dp_grad, new_noise_state = plan.noise_addition_transform.update(
        clipped_grad_sum, state.noise_state
    )
    updates, new_opt_state = optimizer.update(
        dp_grad, state.opt_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        noise_state=new_noise_state,
    )
    return new_state, aux

  step = 0
  for indices in batch_iterator:
    indices = batch_selection.pad_to_multiple_of(indices, padding_multiple)
    batch, is_padding_example = _get_batch(dataset, indices)

    state, aux = step_fn(state, batch, is_padding_example)
    step += 1

    del indices, batch, is_padding_example

    if callback is not None:
      callback(step, state, aux)

  return state
