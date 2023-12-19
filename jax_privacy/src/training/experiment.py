# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

import abc
from typing import Callable, Iterator, Mapping, Protocol

import chex
import jax
from jax_privacy.src.accounting import accountant as accountant_py
from jax_privacy.src.dp_sgd import gradients
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import averaging
from jax_privacy.src.training import devices
from jax_privacy.src.training import dp_updater
from jax_privacy.src.training import evaluator as evaluator_py
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import forward
from jax_privacy.src.training import optimizer_config as opt_config
from jax_privacy.src.training import updater as updater_py
from jaxline import utils as jaxline_utils
import numpy as np


class ExperimentLike(Protocol):
  """Interface for the experiment."""

  def initialize(
      self,
      rng_init: chex.PRNGKey,
  ) -> tuple[
      updater_py.UpdaterState,
      updater_py.StepOnHost,
      Iterator[typing.InputsT],
  ]:
    """Initializes the experiment."""

  @property
  def max_num_updates(self) -> int:
    """Maximum number of update steps."""

  def update(
      self,
      state: updater_py.UpdaterState,
      inputs_producer: Callable[[], typing.InputsT],
      step_on_host: updater_py.StepOnHost,
  ) -> tuple[
      updater_py.UpdaterState,
      typing.NumpyMetrics,
      updater_py.StepOnHost,
  ]:
    """Performs an update step."""

  def evaluate(
      self,
      *,
      state: updater_py.UpdaterState,
      step_on_host: updater_py.StepOnHost,
  ) -> typing.NumpyMetrics:
    """Evaluates on the eval dataset."""


def to_numpy(
    tree: Mapping[str, jax.Array],
) -> typing.NumpyMetrics:
  """Casts the leaves to scalars or list of scalars."""
  # Copy to host (in parallel).
  tree = jax.device_get(tree)
  return jax.tree_map(lambda x: np.squeeze(np.asarray(x)), tree)


def make_experiment_args(
    *,
    rng_train: chex.PRNGKey,
    training_config: experiment_config.TrainingConfig,
    averaging_configs: Mapping[str, averaging.AveragingConfig],
    optimizer_config: opt_config.OptimizerConfig,
    num_training_samples: int,
    forward_fn: forward.ForwardFn,
    device_layout: devices.DeviceLayout = devices.DeviceLayout(),
) -> tuple[dp_updater.Updater, accountant_py.Accountant, int]:
  """Returns the arguments to create a JaxPrivacyExperiment."""

  exact_accountant = training_config.dp.make_accountant(
      batch_size=training_config.batch_size.total,
      batch_size_scale_schedule=training_config.batch_size.scale_schedule,
      num_samples=num_training_samples,
  )
  if training_config.dp.auto_tune_target_epsilon:
    max_num_updates = exact_accountant.compute_max_num_updates(
        epsilon=training_config.dp.auto_tune_target_epsilon,
    )
  else:
    max_num_updates = training_config.num_updates

  if training_config.dp.accountant_cache_num_points > 0:
    accountant = accountant_py.CachedExperimentAccountant(
        max_num_updates=max_num_updates,
        accountant=exact_accountant,
        num_cached_points=training_config.dp.accountant_cache_num_points,
    )
  else:
    accountant = exact_accountant

  grad_computer = gradients.DpsgdGradientComputer(
      clipping_norm=training_config.dp.clipping_norm,
      noise_multiplier=training_config.dp.noise_multiplier,
      rescale_to_unit_norm=training_config.dp.rescale_to_unit_norm,
      vectorize_grad_clipping=training_config.dp.vectorize_grad_clipping,
  )
  updater = dp_updater.Updater(
      batch_size_config=training_config.batch_size,
      forward_fn=forward_fn,
      grad_computer=grad_computer,
      weight_decay=training_config.weight_decay,
      optimizer_config=optimizer_config,
      averaging_configs=averaging_configs,
      max_num_updates=max_num_updates,
      is_trainable=training_config.is_trainable,
      logging_config=training_config.logging,
      rng=rng_train,
      device_layout=device_layout,
      num_training_samples=num_training_samples,
  )

  return updater, accountant, max_num_updates


class JaxPrivacyExperiment(metaclass=abc.ABCMeta):
  """Experiment to perform DP-SGD training."""

  def __init__(
      self,
      *,
      updater: updater_py.AbstractUpdater,
      evaluator: evaluator_py.Evaluator,
      accountant: accountant_py.Accountant,
      num_updates: int,
  ):

    self._max_num_updates = num_updates
    self._accountant = accountant
    self._updater = updater
    self._evaluator = evaluator

  def initialize(
      self,
      rng_init: chex.PRNGKey,
  ) -> tuple[
      updater_py.UpdaterState,
      updater_py.StepOnHost,
      Iterator[typing.InputsT],
  ]:
    """Initializes the experiment."""
    train_input = jaxline_utils.py_prefetch(self.build_train_input)
    state, step_on_host = self._updater.init(
        rng=rng_init,
        inputs=next(train_input),
    )
    return state, step_on_host, train_input

  @property
  def max_num_updates(self) -> int:
    return self._max_num_updates

  def update(
      self,
      state: updater_py.UpdaterState,
      inputs_producer: Callable[[], typing.InputsT],
      step_on_host: updater_py.StepOnHost,
  ) -> tuple[
      updater_py.UpdaterState,
      typing.NumpyMetrics,
      updater_py.StepOnHost,
  ]:
    """Perform a single step of training."""
    state, metrics, step_on_host = (
        self._updater.update(
            state=state,
            inputs_producer=inputs_producer,
            step_on_host=step_on_host,
        )
    )

    scalars = jaxline_utils.get_first(metrics.scalars)
    scalars = to_numpy(scalars)
    scalars['dp_epsilon'] = np.asarray(self._accountant.compute_epsilon(
        num_updates=step_on_host,
        allow_approximate_cache=True,
    ))
    return state, scalars, step_on_host

  @abc.abstractmethod
  def build_train_input(self) -> Iterator[typing.InputsT]:
    """Builds the training input pipeline."""

  @abc.abstractmethod
  def build_eval_input(self) -> Iterator[typing.InputsT]:
    """Builds the training input pipeline."""

  def evaluate(
      self,
      *,
      state: updater_py.UpdaterState,
      step_on_host: updater_py.StepOnHost,
  ) -> typing.NumpyMetrics:
    """Run the complete evaluation with the current model parameters."""
    metrics = self._evaluator.evaluate_dataset(
        updater_state=state,
        ds_iterator=self.build_eval_input(),
    )
    scalars = to_numpy(metrics)
    scalars['update_step'] = np.asarray(step_on_host)
    scalars['dp_epsilon'] = np.asarray(self._accountant.compute_epsilon(
        step_on_host))
    return scalars
