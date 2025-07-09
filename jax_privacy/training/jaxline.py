# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
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

"""Jaxline experiment to define training and eval loops."""

import functools
from typing import Any

import chex
import jax
from jax_privacy.dp_sgd import typing
from jax_privacy.training import experiment as experiment_py
from jax_privacy.training import updater as updater_py
from jaxline import experiment as jaxline_experiment
import ml_collections


class JaxlineWrapper(jaxline_experiment.AbstractExperiment):
  """Wraps an experiment to perform training and evaluation with Jaxline."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_updater_state': 'updater_state',
  }
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      '_step_on_host': 'step_on_host',
  }

  def __init__(
      self,
      mode: str,
      config: Any,
      init_rng: chex.PRNGKey,
      *,
      experiment: experiment_py.ExperimentLike,
      add_prefix_to_scalars: bool = True,
  ):
    """Initializes experiment."""
    del mode, config  # unused
    self._init_rng = init_rng
    self._experiment = experiment
    self._step_on_host: updater_py.StepOnHost | None = None
    self._updater_state: updater_py.UpdaterState | None = None
    self._train_input = None
    self._add_prefix_to_scalars = add_prefix_to_scalars

  def step(
      self,
      *,
      global_step: jax.Array,
      rng: chex.PRNGKey,
      writer: Any,
  ) -> typing.NumpyMetrics:
    """Perform a single step of training."""
    del global_step  # unused
    del writer  # unused
    del rng  # inapplicable as it's per-microbatch; managed instead by updater

    # Check if we need to initialize the training data, and possibly
    # updater_state and step_on_host.
    if self._train_input is None:
      init_updater_state, init_step_on_host, self._train_input = (
          self._experiment.initialize(self._init_rng)
      )
      # Only set updater_state and step_on_host if needed: they might already be
      # set after e.g. a pre-emption.
      if self._updater_state is None:
        self._updater_state = init_updater_state
      if self._step_on_host is None:
        self._step_on_host = init_step_on_host

    self._updater_state, scalars, self._step_on_host = self._experiment.update(
        state=self._updater_state,
        inputs_producer=functools.partial(next, self._train_input),
        step_on_host=self._step_on_host,
    )
    if self._add_prefix_to_scalars:
      scalars = {f'train/{k}': v for k, v in scalars.items()}

    return scalars

  def should_run_step(
      self,
      unused_global_step: int,
      unused_config: ml_collections.ConfigDict,
  ) -> bool:
    """Returns whether to run the step function, given the current update_step.

    We ignore the global_step and config given by jaxline, because model updates
    are not applied at every global_step (due to gradient accumulation to use
    large batch-sizes), so we rather use our own `update_step`, which correctly
    accounts for that.
    """
    if self._step_on_host is None:
      return True
    else:
      return self._step_on_host < self._experiment.max_num_updates

  def evaluate(
      self,
      *,
      global_step: jax.Array,
      rng: chex.PRNGKey,
      writer: Any,
  ) -> typing.NumpyMetrics:
    """Run the complete evaluation with the current model parameters."""
    del global_step  # unused
    del writer  # unused
    del rng  # unused; managed instead by updater
    scalars = self._experiment.evaluate(
        state=self._updater_state,
        step_on_host=self._step_on_host,
    )

    if self._add_prefix_to_scalars:
      scalars = {f'eval/{k}': v for k, v in scalars.items()}
    return scalars
