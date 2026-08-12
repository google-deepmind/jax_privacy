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

This module provides :class:`~jax_privacy.training.DPTrainer`, a class that
encapsulates the static configuration for a DP training loop (execution plan,
loss function, optimizer) and exposes a reusable ``train_step`` that can be
independently JIT-compiled or ahead-of-time compiled.

Design notes: This module aims to provide a framework-agnostic and model
agnostic DP training loop that can be easily configured to use a variety of
mechanism variants and performance knobs. Since it is defined in terms of an
ExecutionPlanConfig object, it automatically works with any mechanism plan
supported by our Tier 2 API.

Because it is designed to be framework and model-agnostic, there are a few
things that are explicitly out of scope, like checkpointing and metrics logging.
Callers are expected to implement these things themselves via dependency
injection through the callback_fn option.
"""

from collections.abc import Callable
import dataclasses
import functools
from typing import Protocol, TypeAlias

from absl import logging
import jax
import jax_privacy
from jax_privacy import _compilation
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import execution_plan
from jax_privacy import optimizers as aug_optimizers
import numpy as np
import optax

# Re-export key symbols so users can access them via jax_privacy.training.
BandMFConfig = execution_plan.BandMFConfig
DPExecutionPlan = execution_plan.DPExecutionPlan
ExecutionPlanConfig = execution_plan.ExecutionPlanConfig
PerformanceFlags = execution_plan.PerformanceFlags

Loss: TypeAlias = jax.Array
Aux: TypeAlias = optax.ArrayTree
PerExampleAux: TypeAlias = jax_privacy.clipping.AuxiliaryOutput
Batch: TypeAlias = optax.ArrayTree
Dataset: TypeAlias = optax.ArrayTree
Params: TypeAlias = optax.ArrayTree
OptState: TypeAlias = optax.ArrayTree
NoiseState: TypeAlias = optax.ArrayTree
# Re-exported so callers can keep using training.PrecompiledFuture.
PrecompiledFuture: TypeAlias = _compilation.PrecompiledFuture
# Compilation strategies, re-exported for the public API.
CompilationStrategy = _compilation.CompilationStrategy
PadToMultiple = _compilation.PadToMultiple
AutotuneMicrobatch = _compilation.AutotuneMicrobatch


class LossFn(Protocol):
  """Expected contract for loss functions used in DP training.

  Loss functions must accept ``params`` and a ``data`` batch, and return
  ``(loss, aux)``.  They may optionally accept a PRNG key as a third
  positional argument for stochastic operations (e.g., dropout).

  Any additional context the loss function needs — frozen parameters,
  model configuration, label smoothing constants, etc. — should be closed
  over before passing the function to :class:`~jax_privacy.training.DPTrainer`::

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingState:
  """Container for the state of the training loop."""

  step: int
  params: Params
  opt_state: OptState
  noise_state: NoiseState


CallbackFn: TypeAlias = Callable[[int, TrainingState, PerExampleAux], None]


def _get_batch(dataset: Batch, indices: np.ndarray) -> tuple[Batch, jax.Array]:
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

  def _index_and_zero(x):
    mask = np.expand_dims(is_padding, tuple(range(1, x.ndim)))
    return jax.device_put(np.where(mask, 0, x[indices]))

  return jax.tree.map(_index_and_zero, dataset), jax.device_put(is_padding)


# DPTrainer contains static configuration that defines the training step, but
# not all fields are hashable (e.g., BandMFConfig contains an `array` field).
# To get jax.jit to work, we use `eq=False` below to hash based on object ID.
# TODO: Refactor to reduce unnecessary recompilations.
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class DPTrainer:
  """Stateless trainer encapsulating the static configuration of a DP loop.

  ``DPTrainer`` separates *configuration* (plan, loss, optimizer) from
  *per-run state* (data, initial params, RNG seed).  This makes the
  ``train_step`` method available as a standalone callable that can be
  compiled or used independently of the training loop.

  **Sharding**: This class does not shard params or data.  For
  multi-device training, provide ``params`` with explicit sharding
  annotations and configure ``spmd_axis_name`` through
  ``performance_flags``.  If data sharding is needed, ``loss_fn``
  should reshard its inputs using sharding-in-types.

  Attributes:
    config: An :class:`~jax_privacy.execution_plan.ExecutionPlanConfig` (e.g.
      :class:`~jax_privacy.execution_plan.BandMFConfig`) specifying the DP
      mechanism.
    performance_flags: Performance-only flags (numerical precision, sharding,
      memory/compute trade-offs) that do not affect the privacy guarantee.
    loss_fn: The per-example loss function.  See
      :class:`~jax_privacy.training.LossFn`.
    optimizer: An
      :class:`~jax_privacy.optimizers.AugmentedGradientTransformation` or a
      plain :class:`optax.GradientTransformation`.
    compilation_strategy: Selects which ``train_step`` programs are compiled for
      training. :class:`~jax_privacy.training.PadToMultiple` (default) pads
      batches to a multiple to bound recompilations;
      :class:`~jax_privacy.training.AutotuneMicrobatch` picks the largest
      microbatch size that fits device memory and compiles once. See
      :class:`~jax_privacy.training.CompilationStrategy`.
  """

  config: execution_plan.ExecutionPlanConfig
  performance_flags: execution_plan.PerformanceFlags = dataclasses.field(
      default_factory=execution_plan.PerformanceFlags
  )
  loss_fn: LossFn
  optimizer: (
      aug_optimizers.AugmentedGradientTransformation
      | optax.GradientTransformation
  )
  compilation_strategy: _compilation.CompilationStrategy = dataclasses.field(
      default_factory=_compilation.PadToMultiple
  )

  def __post_init__(self):
    _ = self.plan  # Build untraced so cached PRNG key isn't a leaked tracer.

  @functools.cached_property
  def plan(self) -> execution_plan.DPExecutionPlan:
    """``DPExecutionPlan`` built from ``config`` and ``performance_flags``."""
    return self.config.make(self.performance_flags)

  def init(self, params: Params) -> TrainingState:
    """Initialize a ``TrainingState`` at step 0."""
    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)
    # jax_privacy will sometimes upcast gradients to a higher dtype for
    # numerical stability, regardless of the dtype of params.
    grads_like = optax.tree.cast(params, self.performance_flags.dtype)
    noise_state = self.plan.noise_addition_transform.init(grads_like)
    return TrainingState(
        step=jax.numpy.zeros((), dtype=jax.numpy.int32),
        params=params,
        opt_state=optimizer.init(grads_like),
        noise_state=noise_state,
    )

  @jax.jit(static_argnames=["self"], donate_argnames=["state"])
  def train_step(
      self,
      state: TrainingState,
      batch: Batch,
      is_padding_example: jax.Array,
      prng_key: jax.Array,
  ) -> tuple[TrainingState, PerExampleAux]:
    """Executes a single DP training step.

    This method is a pure function of its inputs and is safe to wrap with
    ``jax.jit``, ``jax.jit(...).lower()``, or any other JAX transformation.

    Args:
      state: Current ``TrainingState``.
      batch: A PyTree of arrays representing the current mini-batch.
      is_padding_example: A boolean array indicating which examples in ``batch``
        are padding (and should be ignored).
      prng_key: Base PRNG key; a step-specific key is derived via
        ``jax.random.fold_in(prng_key, state.step)``.

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
    loss_prng = jax.random.fold_in(prng_key, state.step)
    clipped_grad_sum, aux = grad_fn(
        state.params, batch, loss_prng, is_padding_example=is_padding_example
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

  def _precompile(
      self,
      dataset: Dataset,
      params: Params,
      *,
      rng_or_seed: np.random.Generator | int | None = None,
  ) -> tuple["DPTrainer", dict[int, PrecompiledFuture]]:
    """[ADVANCED] Resolve config and AOT-compile the steps ``fit`` runs."""
    return _compilation.precompile(
        self, dataset, params, rng_or_seed=rng_or_seed
    )

  def fit(
      self,
      dataset: Dataset,
      state: Params | TrainingState,
      *,
      callback: CallbackFn | None = None,
      rng_or_seed: np.random.Generator | int | None = None,
      precompile: bool = True,
  ) -> TrainingState:
    """Runs an end-to-end differentially private training loop.

    Args:
      dataset: The training dataset, as a PyTree of arrays where the first axis
        of each leaf is the batch / example dimension.
      state: Initial parameter PyTree or a resumable ``TrainingState``. If a
        ``TrainingState`` is provided, training continues from the step recorded
        in the state, and the batch selection iterator safely fast-forwards to
        maintain deterministic DP properties.
      callback: Called after each step as ``callback(step, state, aux)``.
        ``step`` is a Python int.
      rng_or_seed: Optional random seed or ``numpy.random.Generator``, used for
        sampling batches (impacting privacy) and initializing the loss PRNG key
        (potentially impacting utility). Does not influence the noise addition
        transform, which is configured via the
        :class:`~jax_privacy.execution_plan.DPExecutionPlan`.
      precompile: A boolean indicating whether to asynchronously precompile
        ``train_step`` for the batch sizes encountered, instead of just-in-time
        compiling on the fly, which can idle accelerators during training.
        Strategies that resolve before training (e.g. ``AutotuneMicrobatch``)
        run even when this is ``False``.

    Returns:
      Final ``TrainingState``.
    """
    # Precompilation resolves the trainer and compiles the required train_steps.
    # It may return a *different* trainer (e.g. AutotuneMicrobatch picks a
    # concrete microbatch size), so the loop below runs on ``trainer`` rather
    # than ``self``.
    trainer: DPTrainer = self
    futures: dict[int, PrecompiledFuture] = {}

    # Clone state. train_step uses donate_argnames.
    if not isinstance(state, TrainingState):
      state = self.init(jax.tree.map(jax.numpy.copy, state))

    if precompile or isinstance(self.compilation_strategy, AutotuneMicrobatch):
      trainer, futures = self._precompile(
          dataset, state.params, rng_or_seed=rng_or_seed
      )
    assert isinstance(trainer.compilation_strategy, PadToMultiple)
    warn_on_cache_miss = bool(futures)

    # We need tight alignment between how rng is used here and in precompile().
    rng = np.random.default_rng(rng_or_seed)
    prng_key = jax.random.key(int(rng.integers(2**63)))

    num_examples = _validate.batch(dataset)

    with _compilation.hoist_closed_over_constants():
      bss = trainer.plan.batch_selection_strategy
      batch_iterator = bss.batch_iterator(num_examples, rng=rng)

      # initial state.step is 0 (no replay). Otherwise replay to the saved step.
      # TODO: b/545416482 - Investigate stateful iterator instead of rng replay.
      for _ in range(int(state.step)):
        next(batch_iterator)

      step = int(state.step)
      for indices in batch_iterator:
        indices = batch_selection.pad_to_multiple_of(
            indices,
            trainer.compilation_strategy.multiple,
            microbatch_size=trainer.performance_flags.microbatch_size,
        )
        batch, is_padding_example = _get_batch(dataset, indices)
        step_fn = trainer.train_step
        if indices.size in futures:
          step_fn = futures[indices.size].result()
        elif warn_on_cache_miss:
          logging.info(
              "JIT-compiling train_step for batch size %d", indices.size
          )
          logging.warning("Cache Miss! Precompile is not working as intended.")

        state, aux = step_fn(state, batch, is_padding_example, prng_key)
        step += 1

        del indices, batch, is_padding_example

        if callback is not None:
          callback(step, state, aux)

    return state
