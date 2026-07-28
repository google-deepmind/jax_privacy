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

"""Ahead-of-time compilation helpers for the DP training loop.

This module owns the machinery that warms up the ``train_step`` JIT cache by
asynchronously ahead-of-time (AOT) compiling it for the batch sizes a run will
encounter.  It is deliberately kept separate from :mod:`jax_privacy.training`
so the core training loop stays independent of compilation concerns.  The
runtime dependency is one-directional (``training`` imports this module, not the
reverse); the type-only dependency on :class:`~jax_privacy.training.DPTrainer`
is guarded by ``TYPE_CHECKING`` to avoid an import cycle.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import functools
from typing import TYPE_CHECKING, TypeAlias

from absl import logging
import jax
from jax_privacy import _validate
from jax_privacy import batch_selection
import numpy as np

if TYPE_CHECKING:
  from jax_privacy import training  # pylint: disable=g-import-not-at-top

PrecompiledFuture: TypeAlias = concurrent.futures.Future[jax.stages.Compiled]

# Shared thread pool for background ahead-of-time compilation.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# JAX config flag that lowers large closed-over constants as arguments to the
# compiled executable instead of baking them into the HLO as literals.
_SIMPLIFIED_JAXPR_CONSTANTS_FLAG = "jax_use_simplified_jaxpr_constants"


@contextlib.contextmanager
def hoist_closed_over_constants():
  """Lowers large closed-over constants as arguments rather than HLO literals.

  DP fine-tuning loss functions typically close over a large frozen base model
  (this is the documented :class:`~jax_privacy.training.LossFn` contract). By
  default JAX materializes such closed-over arrays into the compiled HLO as
  constants, which inflates compilation time and peak memory, discards input
  sharding, and can OOM for large models. Enabling
  ``jax_use_simplified_jaxpr_constants`` instead hoists them into the compiled
  executable's signature as regular arguments.
  """
  previous = getattr(jax.config, _SIMPLIFIED_JAXPR_CONSTANTS_FLAG)
  jax.config.update(_SIMPLIFIED_JAXPR_CONSTANTS_FLAG, True)
  try:
    yield
  finally:
    jax.config.update(_SIMPLIFIED_JAXPR_CONSTANTS_FLAG, previous)


def precompile(
    trainer: training.DPTrainer,
    dataset: training.Dataset,
    params: training.Params,
    *,
    rng_or_seed: np.random.Generator | int | None = None,
) -> dict[int, PrecompiledFuture]:
  """[ADVANCED] Warm up the JIT cache for ``train_step`` asynchronously."""
  # With the same rng passed to precompile and fit, the exact same
  # batches will be sampled in this dry-run as in the actual training loop,
  # guaranteeing JIT cache hits.
  rng = copy.deepcopy(np.random.default_rng(rng_or_seed))
  seed = rng.integers(2**63)
  n = _validate.batch(dataset)

  state = jax.eval_shape(trainer.init, params)
  key = jax.eval_shape(lambda x: x, jax.random.key(seed))

  futures: dict[int, PrecompiledFuture] = {}

  def _resize(size, x):
    return jax.ShapeDtypeStruct((size, *x.shape[1:]), x.dtype)

  with hoist_closed_over_constants():
    for idx in trainer.plan.batch_selection_strategy.batch_iterator(n, rng=rng):
      padded = batch_selection.pad_to_multiple_of(idx, trainer.padding_multiple)
      batch_size = padded.size
      batch = jax.tree.map(functools.partial(_resize, batch_size), dataset)
      padding = jax.ShapeDtypeStruct((batch_size,), np.bool_)

      lowered = trainer.train_step.lower(trainer, state, batch, padding, key)
      logging.info("AOT-compiling train_step for batch size %d", batch_size)
      # We asyncronously ahead-of-time (AOT) compile the lowered function in a
      # background thread to avoid blocking the training loop. Currently, the
      # rest of this function (batch simulation + lowering) happens on the main
      # thread. This could potentially be improved in the future.
      futures[batch_size] = _COMPILE_POOL.submit(lowered.compile)

  return futures
