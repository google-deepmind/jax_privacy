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

This module provides :class:`DPTrainer`, a class that encapsulates the
static configuration for a DP training loop (execution plan, loss function,
optimizer) and exposes a reusable ``train_step`` that can be independently
JIT-compiled or ahead-of-time compiled.
"""

from collections.abc import Callable
import dataclasses
from typing import Any, Protocol, TypeAlias

import chex
import jax
import jax_privacy
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import execution_plan
from jax_privacy import optimizers as aug_optimizers
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
  over before passing the function to :class:`DPTrainer`::

      frozen = model.freeze(some_params)
      def my_loss(params, data, prng):
          all_params = {**frozen, **params}
          logits = model.apply(all_params, data['x'], rngs={'dropout': prng})
          return cross_entropy(logits, data['y']), {'logits': logits}

      trainer = DPTrainer(..., loss_fn=my_loss, ...)

  NOTE: This signature does not support mutable model state that persists
  across steps (e.g., batch-norm running statistics), as such state is
  generally incompatible with DP-SGD or very difficult to handle correctly.

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class DPTrainer:
  """Stateless trainer encapsulating the static configuration of a DP loop.

  ``DPTrainer`` separates *configuration* (plan, loss, optimizer) from
  *per-run state* (data, initial params, RNG seed).  This makes the
  ``train_step`` method available as a standalone callable that can be
  compiled or used independently of the training loop.

  **Sharding**: This class does not shard params or data.  For
  multi-device training, provide ``params`` with explicit sharding
  annotations and configure ``spmd_axis_name`` through the plan's
  ``PerformanceFlags``.  If data sharding is needed, ``loss_fn``
  should reshard its inputs using sharding-in-types.

  Attributes:
    plan: A ``DPExecutionPlan`` specifying the DP mechanism.
    loss_fn: The per-example loss function.  See :class:`LossFn`.
    optimizer: An ``AugmentedGradientTransformation`` or a plain
      ``optax.GradientTransformation``.
    padding_multiple: If set, batch sizes are padded to a multiple of this
      value.
  """

  plan: execution_plan.DPExecutionPlan
  loss_fn: LossFn
  optimizer: (
      aug_optimizers.AugmentedGradientTransformation
      | optax.GradientTransformation
  )
  padding_multiple: int = 1

  def train_step(
      self,
      state: TrainingState,
      batch: Batch,
      is_padding_example: jax.Array,
      *,
      loss_rng: jax.Array,
  ) -> tuple[TrainingState, PerExampleAux]:
    """Executes a single DP training step.

    This method is a pure function of its inputs and is safe to wrap with
    ``jax.jit``, ``jax.jit(...).lower()``, or any other JAX transformation.

    Args:
      state: Current ``TrainingState``.
      batch: A PyTree of arrays representing the current mini-batch.
      is_padding_example: A boolean array indicating which examples in ``batch``
        are padding (and should be ignored).
      loss_rng: Base PRNG key; a step-specific key is derived via
        ``jax.random.fold_in(loss_rng, state.step)``.

    Returns:
      A tuple ``(new_state, aux)`` where ``new_state`` is the updated
      ``TrainingState`` and ``aux`` is the per-example auxiliary output.
    """
    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)
    pre_clip_fn = optimizer.pre_clipping_transform(state.opt_state)

    grad_fn = self.plan.clipped_grad(
        self.loss_fn,
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

    dp_grad, new_noise_state = self.plan.noise_addition_transform.update(
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

  def fit(
      self,
      dataset: Batch,
      params: Params,
      *,
      callback: (
          Callable[[int, TrainingState, PerExampleAux], None] | None
      ) = None,
      rng: np.random.Generator | int | None = None,
      shard_options: Any = None,
      preload: bool | None = None,
      max_workers: int | None = None,
  ) -> TrainingState:
    """Runs an end-to-end differentially private training loop.

    Args:
      dataset: The training dataset.  This can be either a PyTree of NumPy
        arrays (all data in memory) or a PyGrain ``MapDataset``.  When a
        ``MapDataset`` is provided, PyGrain must be installed; it is not
        required otherwise.
      params: Initial parameter PyTree.
      callback: Called after each step as ``callback(step, state, aux)``.
        ``step`` is a Python int.
      rng: Optional random seed or ``numpy.random.Generator`` for
        reproducibility.
      shard_options: If specified, only a subset of the batch will be loaded
        based on shard_index and shard_count. In multi-controller JAX setups,
        use ``grain.ShardByJaxProcess()`` to have each process load a disjoint
        subset of the batch.  Defaults to no sharding.
      preload: Whether to materialize a PyGrain ``MapDataset`` into host memory
        for fast numpy indexing.  ``True`` forces preloading, ``False`` forces
        streaming, and ``None`` (default) auto-decides based on estimated
        dataset size (preloads if < 1 GiB).
      max_workers: Maximum thread pool workers for concurrent element loading.
        Used in both preload and streaming modes.  Ignored when the dataset is
        an in-memory PyTree.

    Returns:
      Final ``TrainingState``.
    """
    rng = np.random.default_rng(rng)
    loss_rng = jax.random.key(int(rng.integers(2**63)))

    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)

    state = TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        noise_state=self.plan.noise_addition_transform.init(params),
    )

    jit_step = jax.jit(self.train_step)

    # Lazy import: only pull in the data loader when the dataset is a
    # PyGrain MapDataset.  Detection is by class name, not import, so
    # users who don't have grain installed never trigger this path.
    from jax_privacy.experimental import _data_loader  # pylint: disable=g-import-not-at-top,import-outside-toplevel,protected-access

    if _data_loader.is_pygrain_map_dataset(dataset):
      batches = _data_loader.iterate_batches(
          dataset,
          self.plan.batch_selection_strategy,
          rng,
          shard_options=shard_options,
          pad_to_multiple_of=self.padding_multiple,
          preload=preload,
          max_workers=max_workers,
      )
    else:
      num_examples = _validate.batch(dataset)
      batches = self._in_memory_batches(dataset, num_examples, rng)

    step = 0
    for batch, is_padding_example in batches:
      state, aux = jit_step(state, batch, is_padding_example, loss_rng=loss_rng)
      step += 1

      del batch, is_padding_example

      if callback is not None:
        callback(step, state, aux)

    return state

  def _in_memory_batches(self, dataset, num_examples, rng):
    """Yields ``(batch, is_padding)`` tuples from an in-memory PyTree."""
    for indices in self.plan.batch_selection_strategy.batch_iterator(
        num_examples, rng=rng
    ):
      indices = batch_selection.pad_to_multiple_of(
          indices, self.padding_multiple
      )
      yield _get_batch(dataset, indices)
