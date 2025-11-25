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

"""Simple example on MNIST."""

import dataclasses

import haiku as hk
import jax
import image_data
from image_classification import auditing
from image_classification import config_base
from image_classification.models import models
from training import algorithm_config
from training import averaging
from training import experiment_config
from training import optimizer_config
import ml_collections


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class MLPConfig(models.ModelConfig):
  """Configuration for an MLP."""

  def make(self, num_classes: int) -> models.Model:
    def model_fn(images: jax.Array, is_training: bool) -> jax.Array:
      del is_training
      images = hk.Flatten()(images)
      return hk.nets.MLP([16, num_classes])(images)

    return models.Model.from_hk_module(lambda: model_fn)


def get_config() -> ml_collections.ConfigDict:
  """Experiment config."""

  config = config_base.ExperimentConfig(
      optimizer=optimizer_config.sgd_config(
          lr=optimizer_config.constant_lr_config(2.0),
      ),
      model=MLPConfig(),
      training=experiment_config.TrainingConfig(
          num_updates=1_000,
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=8,
              per_device_per_step=8,
          ),
          weight_decay=0.0,  # L-2 regularization,
          dp=experiment_config.DpConfig(
              delta=1e-5,
              clipping_norm=0.1,
              auto_tune_target_epsilon=8.0,
              rescale_to_unit_norm=True,
              algorithm=algorithm_config.DpsgdConfig(noise_multiplier=None),
              auto_tune_field='noise_multiplier',
          ),
          logging=experiment_config.LoggingConfig(),
      ),
      averaging={
          'ema': averaging.ExponentialMovingAveragingConfig(decay=0.999),
      },
      data_train=image_data.MnistLoader(
          config=image_data.MnistTrainConfig(),
      ),
      data_eval=image_data.MnistLoader(
          config=image_data.MnistValidConfig(),
      ),
      evaluation=experiment_config.EvaluationConfig(
          batch_size=100,
      ),
      privacy_auditing=auditing.AuditingConfig(
          canary_type=auditing.CanaryType.MISLLABELLED_INPUT,
          canary_every_n_steps=2,
          num_canaries=100,
      ),
  )

  return config_base.build_jaxline_config(
      experiment_config=config,
  )
