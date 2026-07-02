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
import concurrent.futures
import copy
import dataclasses
import functools
from typing import Protocol, TypeAlias

from absl import logging
import chex
import jax
import jax_privacy
from jax_privacy import _validate
from jax_privacy import batch_selection
from jax_privacy import execution_plan
from jax_privacy import optimizers as aug_optimizers
import numpy as np
import optax


# Re-export key symbols so users can access them via jax_privacy.training.
BandMFConfig = execution_plan.BandMFConfig
DPExecutionPlan = execution_plan.DPExecutionPlan
PerformanceFlags = execution_plan.PerformanceFlags

Loss: TypeAlias = jax.Array
Aux: TypeAlias = chex.ArrayTree
PerExampleAux: TypeAlias = jax_privacy.clipping.AuxiliaryOutput
Batch: TypeAlias = chex.ArrayTree
Dataset: TypeAlias = chex.ArrayTree
Params: TypeAlias = chex.ArrayTree
OptState: TypeAlias = chex.ArrayTree
NoiseState: TypeAlias = chex.ArrayTree
PrecompiledFuture: TypeAlias = concurrent.futures.Future[jax.stages.Compiled]

# Shared thread pool for background ahead-of-time compilation.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


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
    padding_multiple: If set, batch sizes are padded to a multiple of this value
      to limit JIT recompilations from varying Poisson batch sizes.
  """

  plan: execution_plan.DPExecutionPlan
  loss_fn: LossFn
  optimizer: (
      aug_optimizers.AugmentedGradientTransformation
      | optax.GradientTransformation
  )
  padding_multiple: int = 32

  def init(self, params: Params) -> TrainingState:
    """Initialize a ``TrainingState`` at step 0."""
    optimizer = aug_optimizers.as_augmented_optimizer(self.optimizer)
    return TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        noise_state=self.plan.noise_addition_transform.init(params),
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
  ) -> dict[int, PrecompiledFuture]:
    """[ADVANCED] Warm up the JIT cache for ``train_step`` asynchronously."""
    # With the same rng passed to _precompile and fit, the exact same
    # batches will be sampled in this dry-run as in the actual training loop,
    # guaranteeing JIT cache hits.
    rng = copy.deepcopy(np.random.default_rng(rng_or_seed))
    seed = rng.integers(2**63)
    n = _validate.batch(dataset)

    state = jax.eval_shape(self.init, params)
    key = jax.eval_shape(lambda x: x, jax.random.key(seed))

    futures: dict[int, PrecompiledFuture] = {}

    def _resize(size, x):
      return jax.ShapeDtypeStruct((size, *x.shape[1:]), x.dtype)

    for idx in self.plan.batch_selection_strategy.batch_iterator(n, rng=rng):
      padded = batch_selection.pad_to_multiple_of(idx, self.padding_multiple)
      batch_size = padded.size
      batch = jax.tree.map(functools.partial(_resize, batch_size), dataset)
      padding = jax.ShapeDtypeStruct((batch_size,), np.bool_)

      lowered = self.train_step.lower(self, state, batch, padding, key)
      logging.info("AOT-compiling train_step for batch size %d", batch_size)
      # We asyncronously ahead-of-time (AOT) compile the lowered function in a
      # background thread to avoid blocking the training loop. Currently, the
      # rest of this function (batch simulation + lowering) happens on the main
      # thread. This could potentially be improved in the future.
      futures[batch_size] = _COMPILE_POOL.submit(lowered.compile)

    return futures

  def fit(
      self,
      dataset: Dataset,
      params: Params,
      *,
      callback: CallbackFn | None = None,
      rng_or_seed: np.random.Generator | int | None = None,
      precompile: bool = True,
  ) -> TrainingState:
    """Runs an end-to-end differentially private training loop.

    Args:
      dataset: The training dataset, as a PyTree of arrays where the first axis
        of each leaf is the batch / example dimension.
      params: Initial parameter PyTree.
      callback: Called after each step as ``callback(step, state, aux)``.
        ``step`` is a Python int.
      rng_or_seed: Optional random seed or ``numpy.random.Generator``, used for
        sampling batches (impacting privacy) and initializing the loss PRNG key
        (potentially impacting utility). Does not influence the noise addition
        transform, which is configured via the DPExecutionPlan.
      precompile: A boolean indicating whether to asyncronously precompile
        ``train_step`` for the batch sizes encountered, instead of just-in-time
        compiling on the fly, which can idle accelerators during training.

    Returns:
      Final ``TrainingState``.
    """
    futures: dict[int, PrecompiledFuture] = {}
    if precompile:
      futures = self._precompile(dataset, params, rng_or_seed=rng_or_seed)

    # We need tight alignement between how rng is used here and in precompile().
    rng = np.random.default_rng(rng_or_seed)
    prng_key = jax.random.key(int(rng.integers(2**63)))

    num_examples = _validate.batch(dataset)
    # Copy here due to the donate_argnames on the jit decorated train_step.
    state = self.init(jax.tree.map(jax.numpy.copy, params))

    batch_iterator = self.plan.batch_selection_strategy.batch_iterator(
        num_examples, rng=rng
    )

    step = 0
    for indices in batch_iterator:
      indices = batch_selection.pad_to_multiple_of(
          indices, self.padding_multiple
      )
      batch, is_padding_example = _get_batch(dataset, indices)
      if indices.size in futures:
        step_fn = futures[indices.size].result()
      else:
        logging.info("JIT-compiling train_step for batch size %d", indices.size)
        if precompile:
          logging.warning("Cache Miss! Precompile is not working as intended.")
        step_fn = self.train_step

      state, aux = step_fn(state, batch, is_padding_example, prng_key)
      step += 1

      del indices, batch, is_padding_example

      if callback is not None:
        callback(step, state, aux)

    return state
