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

import collections
from typing import Iterable, Iterator

import chex
import haiku as hk
import jax
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import forward
from jax_privacy.experiments.image_classification import models
from jax_privacy.src.training import experiment
from jax_privacy.src.training import metrics as metrics_module


class Experiment(experiment.AbstractExperiment):
  """Jaxline experiment.

  This class controls the training and evaluation loop at a high-level.
  """

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_base.ExperimentConfig,
  ):
    """Initializes experiment.

    Args:
      mode: 'train' or 'eval'.
      init_rng: random number generation key for initialization.
      config: ConfigDict holding all hyper-parameters of the experiment.
    """
    # Unused since we rather rely on `config.random_seed`. The argument
    # `init_rng` is kept to conform to jaxline's expectation.
    del init_rng

    self.config = config

    self._forward_fn = forward.MultiClassForwardFn(
        net=hk.transform_with_state(self._model_fn))

    super().__init__(
        mode=mode,
        random_seed=self.config.random_seed,
        training_config=self.config.training,
        optimizer_config=self.config.optimizer,
        averaging_config=self.config.averaging,
        num_training_samples=self.config.data_train.config.num_samples,
        num_updates=self.config.num_updates,
    )

  @property
  def forward_fn(self) -> forward.MultiClassForwardFn:
    return self._forward_fn

  def _model_fn(self, inputs, is_training=False):
    model_kwargs = {
        'num_classes': self.config.data_train.config.num_classes,
        **self.config.model.kwargs,
    }
    model_instance = models.get_model_instance(self.config.model.name,
                                               model_kwargs)
    return model_instance(
        inputs,
        is_training=is_training,
    )

  def _should_restore_model(self) -> bool:
    return bool(self.config.model.restore.path)

  def _restore_model(self):
    self._params, self._network_state = models.restore_from_path(
        restore_path=self.config.model.restore.path,
        params_key=self.config.model.restore.params_key,
        network_state_key=self.config.model.restore.network_state_key,
        layer_to_reset=self.config.model.restore.layer_to_reset,
        params_init=self._params,
        network_state_init=self._network_state,
    )

  def _build_train_input(self) -> Iterator[data.DataInputs]:
    """Builds the training input pipeline."""
    return self.config.data_train.load_dataset(
        batch_dims=(
            jax.local_device_count(),
            self.batching.batch_size_per_device_per_step,
        ),
        is_training=True,
        shard_data=True,
    )

  def _build_eval_input(self) -> Iterator[data.DataInputs]:
    """Builds the evaluation input pipeline."""
    return self.config.data_eval.load_dataset(
        batch_dims=(
            jax.process_count(),
            jax.local_device_count(),
            self.config.evaluation.batch_size,
        ),
        is_training=False,
        shard_data=False,
        max_num_batches=self.config.evaluation.max_num_batches,
    )

  def _eval_epoch(self, rng, unused_global_step):
    """Evaluates an epoch."""
    avg_metrics = collections.defaultdict(metrics_module.Avg)

    # Checkpoints broadcast for each local device, which we undo here since the
    # evaluation is performed on a single device (it is not pmapped).
    if isinstance(self._averaging_config.ema_coefficient, Iterable):
      ema_params = {
          f'ema_{ema_decay}': params_ema for ema_decay, params_ema in zip(
              self._averaging_config.ema_coefficient,
              self._params_ema,
              strict=True)
      }
    elif self._params_ema is not None:
      ema_params = {'ema': self._params_ema}
    else:
      ema_params = {}
    if self._params_polyak is not None:
      polyak_params = {'polyak': self._params_polyak}
    else:
      polyak_params = {}
    params_dict = {
        'last': self._params,
        **ema_params,
        **polyak_params,
    }

    state = self._network_state
    num_samples = 0
    host_id = jax.process_index()

    # Iterate over the evaluation dataset and accumulate the metrics.
    for inputs in self._build_eval_input():
      rng, rng_eval = jax.random.split(rng)
      num_hosts, num_devices_per_host, batch_size_per_device, *_ = (
          inputs.image.shape)
      batch_size = num_hosts * num_devices_per_host * batch_size_per_device
      num_samples += batch_size
      local_inputs = jax.tree_map(lambda x: x[host_id], inputs)

      # Evaluate batch for each set of parameters.
      for params_name, params in params_dict.items():
        metrics = self.updater.evaluate(params, state, rng_eval, local_inputs)

        # Update accumulated average for each metric.
        for metric_name, val in metrics.scalars.items():
          avg_metrics[f'{metric_name}_{params_name}'].update(val, n=batch_size)

    metrics = {k: v.avg for k, v in avg_metrics.items()}
    metrics['num_samples'] = num_samples

    return metrics
