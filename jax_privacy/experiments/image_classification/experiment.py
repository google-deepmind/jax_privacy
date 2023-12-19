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

from typing import Iterator

from absl import logging
import jax
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import evaluator as evaluator_py
from jax_privacy.experiments.image_classification import forward
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import experiment
from jax_privacy.src.training import updater as updater_py
from jaxline import utils as jaxline_utils
import numpy as np


class ImageClassificationExperiment(experiment.JaxPrivacyExperiment):
  """Jaxline experiment.

  This class controls the training and evaluation loop at a high-level.
  """

  def __init__(self, config: config_base.ExperimentConfig):
    net = config.model.make(config.data_train.config.num_classes)
    if isinstance(config.data_train.config, data.MULTILABEL_DATASETS):
      assert isinstance(config.data_eval.config, data.MULTILABEL_DATASETS)
      assert (config.data_eval.config.class_names
              == config.data_train.config.class_names)
      forward_fn = forward.MultiClassMultiLabelForwardFn(
          net=net,
          all_label_names=config.data_eval.config.class_names,
          class_indices_for_eval=(
              config.data_eval.config.indices_in_select_label_order),
      )
      eval_auc = True
    else:
      forward_fn = forward.MultiClassSingleLabelForwardFn(
          net=net,
          label_smoothing=config.label_smoothing,
      )
      eval_auc = False

    rng = jax.random.PRNGKey(config.random_seed)
    rng_train, rng_eval = jax.random.split(rng)

    dp_updater, accountant, num_updates = (
        experiment.make_experiment_args(
            rng_train=rng_train,
            training_config=config.training,
            optimizer_config=config.optimizer,
            averaging_configs=config.averaging,
            num_training_samples=config.data_train.config.num_samples,
            forward_fn=forward_fn,
        )
    )
    super().__init__(
        updater=dp_updater,
        accountant=accountant,
        num_updates=num_updates,
        evaluator=evaluator_py.ImageClassificationEvaluator(
            forward_fn=forward_fn,
            rng=rng_eval,
            eval_auc=eval_auc,
            eval_disparity=config.eval_disparity,
            class_names=config.data_eval.config.class_names,
        ),
    )
    self._config = config

  def build_train_input(self) -> Iterator[data.DataInputs]:
    """Builds the training input pipeline."""
    return self._config.data_train.load_dataset(
        batch_dims=(
            jax.local_device_count(),
            self._config.training.batch_size.per_device_per_step,
        ),
        is_training=True,
        shard_data=True,
    )

  def build_eval_input(
      self,
      loader: data.DataLoader | None = None,
  ) -> Iterator[data.DataInputs]:
    """Builds the evaluation input pipeline."""
    if not loader:
      loader = self._config.data_eval
    return loader.load_dataset(
        batch_dims=(
            jax.process_count(),
            jax.local_device_count(),
            self._config.evaluation.batch_size,
        ),
        is_training=False,
        shard_data=False,
        max_num_batches=self._config.evaluation.max_num_batches,
    )

  def evaluate(
      self,
      *,
      state: updater_py.UpdaterState,
      step_on_host: updater_py.StepOnHost,
  ) -> typing.NumpyMetrics:
    """Run the complete evaluation with the current model parameters."""
    dp_epsilon = self._accountant.compute_epsilon(step_on_host)

    if self._config.data_eval_additional:
      loaders = [None, self._config.data_eval_additional]
      prefixes = ['', 'addeval_']
    else:
      loaders = [None]
      prefixes = ['']

    metrics = {}
    for loader, prefix in zip(loaders, prefixes, strict=True):
      loader_metrics = self._evaluator.evaluate_dataset(
          updater_state=state,
          ds_iterator=self.build_eval_input(loader),
      )
      metrics.update({prefix + k: v for k, v in loader_metrics.items()})

    # Convert arrays to scalars for logging and storing.
    metrics = experiment.to_numpy(metrics)

    metrics.update(
        update_step=np.asarray(step_on_host),
        dp_epsilon=np.asarray(dp_epsilon),
    )

    logging.info(metrics)

    # Make sure all hosts stay up until the end of the evaluation.
    jaxline_utils.rendezvous()
    return metrics
