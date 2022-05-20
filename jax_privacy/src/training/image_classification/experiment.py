# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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
from typing import Any, Dict, Union

from absl import flags
from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.src import accounting
from jax_privacy.src.training import averaging
from jax_privacy.src.training import batching
from jax_privacy.src.training import metrics as metrics_module
from jax_privacy.src.training.image_classification import data
from jax_privacy.src.training.image_classification import forward
from jax_privacy.src.training.image_classification import models
from jax_privacy.src.training.image_classification import updater
from jaxline import experiment
from jaxline import utils
import ml_collections
import numpy as np


FLAGS = flags.FLAGS


def _to_scalar(
    x: Union[chex.Numeric, chex.Array, chex.ArrayNumpy],
) -> chex.Scalar:
  """Casts the single-item input to a scalar if it is an array."""
  if isinstance(x, (chex.Array, chex.ArrayNumpy)):
    return x.item()
  else:
    return x


class Experiment(experiment.AbstractExperiment):
  """Jaxline experiment.

  This class controls the training and evaluation loop at a high-level.
  """

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assume that these are all sharded
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
      init_rng: chex.PRNGKey,
      config: ml_collections.ConfigDict,
  ):
    """Initializes experiment.

    Args:
      mode: 'train' or 'eval'.
      init_rng: random number generation key for initialization.
      config: ConfigDict holding all hyper-parameters of the experiment.
    """

    super().__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    self._params = None
    self._network_state = None
    self._opt_state = None

    self._params_ema = None
    self._params_polyak = None

    self._train_input = None
    self._eval_input = None

    self.net = hk.transform_with_state(self._model_fn)
    self._average_ema = jax.pmap(averaging.ema, axis_name='i')
    self._average_polyak = jax.pmap(averaging.polyak, axis_name='i')

    self.forward_fn = forward.MultiClassForwardFn(net=self.net)

    train_init = self.forward_fn.train_init
    train_forward = self.forward_fn.train_forward
    self._eval_forward = jax.jit(self.forward_fn.eval_forward)

    self._num_classes = self.config.data.dataset.num_classes
    self.num_training_samples = self.config.data.dataset.train.num_samples

    cfg_batch_size = self.config.training.batch_size
    if cfg_batch_size.scale_schedule:
      # cfg_batch_size.scale_schedule is specified as a Mapping[str, float]
      # (rather than Mapping[int, float]) in order to work well with
      # ConfigDicts, so we manually cast the keys to integers
      scale_schedule = {
          int(k): v for k, v in cfg_batch_size.scale_schedule.items()
      }
    else:
      scale_schedule = None

    self.batching = batching.VirtualBatching(
        batch_size_init=cfg_batch_size.init_value,
        batch_size_per_device_per_step=cfg_batch_size.per_device_per_step,
        scale_schedule=scale_schedule,
    )

    self.accountant = accounting.Accountant(
        clipping_norm=self.config.training.dp.clipping_norm,
        std_relative=self.config.training.dp.noise.std_relative,
        dp_epsilon=self.config.training.dp.stop_training_at_epsilon,
        dp_delta=self.config.training.dp.target_delta,
        batching=self.batching,
        num_samples=self.num_training_samples,
    )

    if self.config.training.dp.stop_training_at_epsilon:
      self._max_num_updates = self.accountant.compute_max_num_updates()
    else:
      self._max_num_updates = self.config.num_updates

    # When a keyword argument is specified in `relative_schedule_kwargs`, it
    # means that its schedule is defined only relatively to the total number
    # of model updates. We now adapt that value to rather be specified in
    # absolute terms so that it can be correctly interpreted by optax.
    if self.config.optimizer.lr.relative_schedule_kwargs is not None:
      for kwarg_name in self.config.optimizer.lr.relative_schedule_kwargs:
        rel_val = self.config.optimizer.lr.decay_schedule_kwargs[kwarg_name]
        abs_val = rel_val * self._max_num_updates
        self.config.optimizer.lr.decay_schedule_kwargs[kwarg_name] = abs_val

    self.updater = updater.Updater(
        batching=self.batching,
        train_init=train_init,
        forward=train_forward,
        clipping_norm=self.config.training.dp.clipping_norm,
        noise_std_relative=self.config.training.dp.noise.std_relative,
        rescale_to_unit_norm=self.config.training.dp.rescale_to_unit_norm,
        weight_decay=self.config.training.weight_decay,
        optimizer_name=self.config.optimizer.name,
        optimizer_kwargs=self.config.optimizer.kwargs,
        lr_init_value=self.config.optimizer.lr.init_value,
        lr_decay_schedule_name=self.config.optimizer.lr.decay_schedule_name,
        lr_decay_schedule_kwargs=self.config.optimizer.lr.decay_schedule_kwargs,
        train_only_layer=self.config.training.train_only_layer,
        log_snr_global=self.config.training.logging.snr_global,
        log_snr_per_layer=self.config.training.logging.snr_per_layer,
        log_grad_clipping=self.config.training.logging.grad_clipping,
        log_grad_alignment=self.config.training.logging.grad_alignment,
    )

  def _compute_epsilon(self, num_updates: chex.Numeric) -> float:
    if jnp.size(num_updates) > 0:
      num_updates = jnp.reshape(num_updates, [-1])[0]
    return self.accountant.compute_current_epsilon(int(num_updates))

  def _model_fn(self, inputs, is_training=False):
    with self.config.model.model_kwargs.unlocked():
      self.config.model.model_kwargs['num_classes'] = self._num_classes
    model_instance = models.get_model_instance(self.config.model.model_type,
                                               self.config.model.model_kwargs)
    return model_instance(
        inputs,
        is_training=is_training,
    )

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
  ) -> Dict[str, np.ndarray]:
    """Perform a single step of training."""
    del writer  # unused

    if self._train_input is None:
      self._initialize_train()

    self._params, self._network_state, self._opt_state, scalars = (
        self.updater.update(
            params=self._params,
            network_state=self._network_state,
            opt_state=self._opt_state,
            global_step=global_step,
            inputs=next(self._train_input),
            rng=rng,
        ))

    # Just return the tracking metrics on the first device for logging.
    scalars = utils.get_first(scalars)
    self._average_params()

    # Calculating these scalars at each step leads to a drastic reduction in TPU
    # duty cycle (from 90% to 50%).
    if self.update_step % 100 == 0:
      # Log dp_epsilon (outside the pmapped _update_func method).
      scalars.update(dp_epsilon=self._compute_epsilon(scalars['update_step']))

    # Convert arrays to scalars for logging and storing.
    return jax.tree_map(_to_scalar, scalars)

  def _average_params(self):
    """Performs both EMA and Polyak parameter averaging."""

    t = utils.bcast_local_devices(self.update_step)
    mu = utils.bcast_local_devices(self.config.averaging.ema.coefficient)

    self._params_ema = self._average_ema(
        tree_old=self._params_ema,
        tree_new=self._params,
        mu=mu,
        t=t-self.config.averaging.ema.start_step,
    )

    self._params_polyak = self._average_polyak(
        tree_old=self._params_polyak,
        tree_new=self._params,
        t=t-self.config.averaging.polyak.start_step,
    )

  @property
  def update_step(self):
    """Number of model updates performed so far."""
    if self._opt_state is not None:
      # `update step` is logged in the optimizer state (by optax.MultiSteps)
      # under the name of 'gradient_step'.
      return utils.get_first(self._opt_state.gradient_step)
    else:
      # The optimizer state is not initialized yet, thus no step has been taken
      # so far.
      return 0

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
    return self.update_step < self._max_num_updates

  def _initialize_train(self):
    """Initializes the data pipeline, the model and the optimizer."""
    self._train_input = utils.py_prefetch(self._build_train_input)

    # Check that params have not already been restored.
    if self._params is None:
      rng_key = utils.bcast_local_devices(self.init_rng)

      self._params, self._network_state, self._opt_state = self.updater.init(
          inputs=next(self._train_input), rng_key=rng_key)

      if self.config.model.restore.path:
        self._params, self._network_state = models.restore_from_path(
            restore_path=self.config.model.restore.path,
            params_key=self.config.model.restore.params_key,
            network_state_key=self.config.model.restore.network_state_key,
            layer_to_reset=self.config.model.restore.layer_to_reset,
            params_init=self._params,
            network_state_init=self._network_state,
        )
        logging.info('Initialized parameters from a checkpoint.')
      else:
        logging.info('Initialized parameters randomly rather than restoring '
                     'from checkpoint.')

      self._params_ema = self._params
      self._params_polyak = self._params

  def _build_train_input(self):
    """Builds the training input pipeline."""
    bs_per_device_per_step = self.batching.batch_size_per_device_per_step
    image_size_train = self.config.data.get('image_size.train')
    return data.build_train_input(
        dataset=self.config.data.dataset,
        image_size_train=image_size_train,
        augmult=self.config.data.augmult,
        random_crop=self.config.data.random_crop,
        random_flip=self.config.data.random_flip,
        batch_size_per_device_per_step=bs_per_device_per_step,
    )

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
  ) -> Dict[str, np.ndarray]:
    """Run the complete evaluation with the current model parameters."""
    del writer  # unused

    dp_epsilon = self._compute_epsilon(self.update_step)

    metrics = jax.tree_map(np.asarray, self._eval_epoch(utils.get_first(rng)))
    metrics.update(
        update_step=self.update_step,
        dp_epsilon=dp_epsilon,
    )

    # Convert arrays to scalars for logging and storing.
    metrics = jax.tree_map(_to_scalar, metrics)

    logging.info(metrics)
    return metrics

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""
    image_size_eval = self.config.data.get('image_size.eval')
    return data.build_eval_input(
        dataset=self.config.data.dataset,
        image_size_eval=image_size_eval,
        batch_size_eval=self.config.evaluation.batch_size,
    )

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    avg_metrics = collections.defaultdict(metrics_module.Avg)

    # Checkpoints broadcast for each local device, which we undo here since the
    # evaluation is performed on a single device (it is not pmapped).
    params_dict = {
        'last': utils.get_first(self._params),
        'ema': utils.get_first(self._params_ema),
        'polyak': utils.get_first(self._params_polyak),
    }

    state = utils.get_first(self._network_state)
    num_samples = 0

    # Iterate over the evaluation dataset and accumulate the metrics.
    for inputs in self._build_eval_input():
      rng, _ = jax.random.split(rng)
      batch_size = len(jax.tree_leaves(inputs)[0])
      num_samples += batch_size

      # Evaluate batch for each set of parameters.
      for params_name, params in params_dict.items():
        unused_logits, batch_metrics = self._eval_forward(
            params, inputs, state, rng)

        # Update accumulated average for each metric.
        for metric_name, val in batch_metrics.items():
          avg_metrics[f'{metric_name}_{params_name}'].update(val, n=batch_size)

    metrics = {k: v.avg for k, v in avg_metrics.items()}
    metrics['num_samples'] = num_samples

    return metrics
