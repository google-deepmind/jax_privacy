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

"""Fine-tuning a WRN-28-10 on CIFAR-100 with (1.0, 1e-5)-DP."""

from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification.models import models
from jax_privacy.src.training import averaging
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Experiment config."""

  config = config_base.ExperimentConfig(
      optimizer=optimizer_config.sgd_config(
          lr=optimizer_config.constant_lr_config(1.0),
      ),
      model=models.WithRestoreModelConfig(
          path=models.Registry.WRN_28_10_IMAGENET32.path,
          params_key='params',
          network_state_key='network_state',
          layer_to_ignore='wide_res_net/Softmax',
          model=models.WideResNetConfig(
              depth=28,
              width=10,
          ),
      ),
      training=experiment_config.TrainingConfig(
          num_updates=250,
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=16384,
              per_device_per_step=16,
          ),
          weight_decay=0.0,
          train_only_layer=None,  # 'wide_res_net/Softmax',
          dp=experiment_config.DPConfig(
              delta=1e-5,
              clipping_norm=1.0,
              auto_tune_target_epsilon=1.0,
              rescale_to_unit_norm=True,
              noise_multiplier=21.1,
              auto_tune_field=None,  # 'num_updates',
          ),
          logging=experiment_config.LoggingConfig(
              grad_clipping=True,
              snr_global=True,  # signal-to-noise ratio across layers
              snr_per_layer=False,  # signal-to-noise ratio per layer
          ),
      ),
      averaging={'ema': averaging.ExponentialMovingAveragingConfig(decay=0.9)},
      data_train=image_data.Cifar100Loader(
          config=image_data.Cifar100TrainValidConfig(
              preprocess_name='standardise',
          ),
          augmult_config=image_data.AugmultConfig(
              augmult=16,
              random_flip=True,
              random_crop=True,
              random_color=False,
          ),
      ),
      data_eval=image_data.Cifar100Loader(
          config=image_data.Cifar100TestConfig(
              preprocess_name='standardise',
          ),
      ),
      evaluation=experiment_config.EvaluationConfig(
          batch_size=100,
      ),
  )

  return config_base.build_jaxline_config(
      experiment_config=config,
  )
