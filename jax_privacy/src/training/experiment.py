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
import functools
from typing import Any, Union

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jax_privacy.src import accounting
from jax_privacy.src.dp_sgd import batching as batching_module
from jax_privacy.src.dp_sgd import devices
from jax_privacy.src.dp_sgd import gradients
from jax_privacy.src.training import dp_updater
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import forward
from jax_privacy.src.training import optimizer_config as opt_config
from jax_privacy.src.training import updater as updater_py
from jaxline import experiment
from jaxline import utils as jaxline_utils
import ml_collections
import numpy as np


def _to_scalar(
    x: Union[chex.Numeric, chex.ArrayNumpy],
) -> Union[chex.Numeric, chex.ArrayNumpy]:
  """Casts the input to a scalar if it is an array with a single element."""
  if isinstance(x, (chex.Array, chex.ArrayNumpy)) and x.size == 1:
    return x.reshape(())
  else:
    return x


class AbstractExperiment(experiment.AbstractExperiment, metaclass=abc.ABCMeta):
  """Jaxline Experiment performing DP-SGD training."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
      '_network_state': 'network_state',
      '_params_ema': 'params_ema',
      '_params_polyak': 'params_polyak',
  }

  def __init__(
      self,
      mode: str,
      random_seed: int,
      training_config: experiment_config.TrainingConfig,
      averaging_config: experiment_config.AveragingConfig,
      optimizer_config: opt_config.OptimizerConfig,
      num_training_samples: int,
      num_updates: int,
      *,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
  ):
    """Initializes experiment."""

    self.mode = mode
    self.random_seed = random_seed
    self._training_config = training_config
    self._averaging_config = averaging_config
    self._optimizer_config = optimizer_config
    self._device_layout = device_layout

    self._params = None
    self._network_state = None
    self._opt_state = None
    self._step_count = updater_py.StepCount(update_step=0, accumulation_step=0)

    # The ema coefficient may be a scalar or a list of scalars.
    self._params_ema = jax.tree_map(lambda _: None,
                                    self._averaging_config.ema_coefficient)
    self._params_polyak = None

    self._train_input = None
    self._eval_input = None

    self.num_training_samples = num_training_samples

    self.batching = batching_module.VirtualBatching(
        batch_size_init=self._training_config.batch_size.total,
        batch_size_per_device_per_step=(
            self._training_config.batch_size.per_device_per_step),
        scale_schedule=self._training_config.batch_size.scale_schedule,
    )

    self.accountant = accounting.ExperimentAccountant(
        clipping_norm=self._training_config.dp.clipping_norm,
        noise_multiplier=self._training_config.dp.noise_multiplier,
        dp_epsilon=self._training_config.dp.stop_training_at_epsilon,
        dp_delta=self._training_config.dp.delta,
        batching=self.batching,
        num_samples=self.num_training_samples,
        dp_accountant_config=self._training_config.dp.accountant,
    )

    if self._training_config.dp.stop_training_at_epsilon:
      self._max_num_updates = self.accountant.compute_max_num_updates()
    else:
      self._max_num_updates = num_updates

    self._cached_accountant = accounting.CachedExperimentAccountant(
        max_num_updates=self._max_num_updates,
        accountant=self.accountant,
    )

    self._updater = None

  @property
  def updater(self) -> updater_py.AbstractUpdater:
    if self._updater is None:
      self._updater = self._build_updater()
    return self._updater

  @property
  @abc.abstractmethod
  def forward_fn(self) -> forward.ForwardFn:
    """Forward function."""

  def _build_updater(self) -> updater_py.AbstractUpdater:
    """Builds a 'standard' Updater from the config."""
    grad_computer = gradients.GradientComputer(
        clipping_norm=self._training_config.dp.clipping_norm,
        noise_multiplier=self._training_config.dp.noise_multiplier,
        rescale_to_unit_norm=self._training_config.dp.rescale_to_unit_norm,
        vectorize_grad_clipping=(
            self._training_config.dp.vectorize_grad_clipping),
        device_layout=self._device_layout,
    )

    return dp_updater.Updater(
        batching=self.batching,
        forward_fn=self.forward_fn,
        grad_computer=grad_computer,
        weight_decay=self._training_config.weight_decay,
        optimizer_config=self._optimizer_config,
        max_num_updates=self._max_num_updates,
        is_trainable=self._training_config.is_trainable,
        logging_config=self._training_config.logging,
        rng_seed=self.random_seed,
        device_layout=self._device_layout,
    )

  def _compute_epsilon(
      self,
      num_updates: chex.Numeric,
      use_approximate_cache: bool = False,
  ) -> float:
    """Computes DP-epsilon either on-the-fly or reusing cached results."""
    if jnp.size(num_updates) > 0:
      num_updates = jnp.reshape(num_updates, [-1])[0]
    num_updates = int(num_updates)
    if use_approximate_cache:
      return self._cached_accountant.compute_approximate_epsilon(num_updates)
    else:
      return self.accountant.compute_current_epsilon(num_updates)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(
      self,
      *,
      global_step: chex.Array,
      rng: chex.Array,
      writer: Any,
  ) -> dict[str, np.ndarray]:
    """Perform a single step of training."""
    del writer  # unused

    if self._train_input is None:
      self._initialize_train()

    (
        self._params,
        self._network_state,
        self._opt_state,
        self._step_count,
        metrics,
    ) = (
        self.updater.update(
            params=self._params,
            network_state=self._network_state,
            opt_state=self._opt_state,
            step_count=self._step_count,
            inputs=next(self._train_input),
        ))

    # Just return the tracking metrics on the first device for logging.
    scalars = jaxline_utils.get_first(metrics.scalars)

    if self._averaging_config.ema_enabled:
      def single_ema(ema_coefficient, params_ema):
        return self.updater.update_ema(
            params_ema, self._params, self._opt_state,
            mu=ema_coefficient,
            start_step=self._averaging_config.ema_start_step,
        )
      self._params_ema = jax.tree_map(
          single_ema, self._averaging_config.ema_coefficient, self._params_ema)

    if self._averaging_config.polyak_enabled:
      self._params_polyak = self.updater.update_polyak(
          self._params_polyak, self._params, self._opt_state,
          start_step=self._averaging_config.polyak_start_step,
      )

    # Convert the number of samples seen into epochs.
    scalars['data_seen'] = self.batching.data_seen(global_step[0])
    scalars['epochs'] = scalars['data_seen'] / self.num_training_samples

    # Log dp_epsilon (outside the pmapped _update_func method).
    scalars.update(dp_epsilon=self._compute_epsilon(
        num_updates=self._step_count.update_step,
        use_approximate_cache=True,
    ))

    if self._training_config.logging.prepend_split_name:
      scalars = {f'train/{k}': v for k, v in scalars.items()}

    # Convert arrays to scalars for logging and storing.
    return jax.tree_util.tree_map(_to_scalar, scalars)

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
    return self._step_count.update_step < self._max_num_updates

  def _initialize_train(self):
    """Initializes the training data and the model parameters."""
    self._train_input = jaxline_utils.py_prefetch(self._build_train_input)

    # Check we haven't already restored params
    if self._params is None:
      rng_init = jax.random.PRNGKey(self.random_seed)
      (
          self._params,
          self._network_state,
          self._opt_state,
          self._step_count,
      ) = self.updater.init(
          rng=rng_init,
          inputs=next(self._train_input),
      )

      if self._should_restore_model():
        # Update self._params and self._network_state
        self._restore_model()
        logging.info('Initialized parameters from a checkpoint.')
      else:
        logging.info('Initialized parameters randomly rather than restoring '
                     'from checkpoint.')

      # Take physical copies of the initial params, so that they remain intact
      # after the first update step when `params` is donated.
      if self._averaging_config.ema_enabled:
        self._params_ema = jax.tree_map(
            lambda _: self.updater.init_average(self._params),
            self._averaging_config.ema_coefficient)
      if self._averaging_config.polyak_enabled:
        self._params_polyak = self.updater.init_average(self._params)

    else:
      # We have restored from a checkpoint. The step count is not in the
      # checkpoint, but is derived as needed from the optimiser state.
      self._step_count = self.updater.step_count_from_opt_state(
          self._opt_state)

  @abc.abstractmethod
  def _should_restore_model(self) -> bool:
    """Whether the model should be restored (or randomly initialized)."""

  @abc.abstractmethod
  def _restore_model(self):
    """Restore model from pre-trained checkpoint."""

  @abc.abstractmethod
  def _build_train_input(self):
    """Builds the training input pipeline."""

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(
      self,
      *,
      global_step: chex.Array,
      rng: chex.Array,
      writer: Any,
  ) -> dict[str, np.ndarray]:
    """Run the complete evaluation with the current model parameters."""
    del writer  # unused

    if self._opt_state is not None:
      # We have restored from a checkpoint. The step count is not in the
      # checkpoint, but is derived as needed from the optimiser state.
      self._step_count = self.updater.step_count_from_opt_state(
          self._opt_state)

    # Ensure that the random key is the same across all hosts.
    rng = jax.pmap(
        functools.partial(jax.lax.all_gather, axis_name='i'),
        axis_name='i')(rng)[:, 0]

    dp_epsilon = self._compute_epsilon(self._step_count.update_step)

    metrics = jax.tree_util.tree_map(
        np.asarray,
        self._eval_epoch(
            jaxline_utils.get_first(rng),
            jaxline_utils.get_first(global_step),
        ),
    )
    metrics.update(
        update_step=self._step_count.update_step,
        dp_epsilon=dp_epsilon,
    )

    if self._training_config.logging.prepend_split_name:
      metrics = {f'eval/{k}': v for k, v in metrics.items()}

    # Convert arrays to scalars for logging and storing.
    metrics = jax.tree_util.tree_map(_to_scalar, metrics)

    logging.info(metrics)

    # Make sure all hosts stay up until the end of the evaluation.
    jaxline_utils.rendezvous()
    return metrics

  @abc.abstractmethod
  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""

  @abc.abstractmethod
  def _eval_epoch(self, rng, global_step):
    """Evaluates an epoch."""
