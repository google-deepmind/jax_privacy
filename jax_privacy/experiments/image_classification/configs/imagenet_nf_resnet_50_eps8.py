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

"""Training an NF-ResNet-50 on ImageNet with (8.0, 8e-7)-DP."""

import haiku.initializers as hk_init
import jax.numpy as jnp
from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Experiment config."""
  config = config_base.ExperimentConfig(
      num_updates=71589,
      optimizer=optimizer_config.sgd_config(
          lr=optimizer_config.constant_lr_config(4.0),
      ),
      model=config_base.ModelConfig(
          name='nf_resnet',
          kwargs={
              'variant': 'ResNet50',
              'drop_rate': None,  # dropout-rate
              'fc_init': hk_init.RandomNormal(0.01, 0),
              'skipinit_gain': jnp.ones,
          },
      ),
      training=experiment_config.TrainingConfig(
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=16384,
              per_device_per_step=32,
          ),
          weight_decay=0.0,  # L-2 regularization,
          train_only_layer=None,  # None
          dp=experiment_config.DPConfig(
              delta=8e-7,
              clipping_norm=1.0,
              stop_training_at_epsilon=8.0,
              rescale_to_unit_norm=True,
              noise_multiplier=2.5,
          ),
          logging=experiment_config.LoggingConfig(
              grad_clipping=True,
              grad_alignment=False,
              snr_global=True,  # signal-to-noise ratio across layers
              snr_per_layer=False,  # signal-to-noise ratio per layer
          ),
      ),
      averaging=experiment_config.AveragingConfig(ema_coefficient=0.99999,),
      data_train=image_data.ImageNetLoader(
          config=image_data.ImagenetTrainConfig(
              preprocess_name='standardise',
              image_size=(224, 224),
          ),
          augmult_config=image_data.AugmultConfig(
              augmult=4,
              random_flip=True,
              random_crop=True,
              random_color=False,
          ),
      ),
      data_eval=image_data.ImageNetLoader(
          config=image_data.ImagenetValidConfig(
              preprocess_name='standardise',
              image_size=(224, 224),
          ),
      ),
      evaluation=experiment_config.EvaluationConfig(
          batch_size=100,
      ),
  )

  return config_base.build_jaxline_config(
      experiment_config=config,
  )
