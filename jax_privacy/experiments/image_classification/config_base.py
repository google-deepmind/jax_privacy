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

"""Base configuration."""

import dataclasses
import random
from typing import Any, Mapping

from jax_privacy.experiments import image_data as data
from jax_privacy.src.training import auto_tune
from jax_privacy.src.training import experiment_config as experiment_config_py
from jax_privacy.src.training import optimizer_config
from jaxline import base_config as jaxline_base_config
import ml_collections


MODEL_CKPT = ml_collections.FrozenConfigDict({
    'WRN_40_4_CIFAR100': 'WRN_40_4_CIFAR100.dill',
    'WRN_40_4_IMAGENET32': 'WRN_40_4_IMAGENET32.dill',
    'WRN_28_10_IMAGENET32': 'WRN_28_10_IMAGENET32.dill',
})


@dataclasses.dataclass(kw_only=True, slots=True)
class ModelRestoreConfig:
  """Configuration for restoring the model.

  Attributes:
    path: Path to the model to restore.
    params_key: (dictionary) Key identifying the parameters in the checkpoint to
      restore.
    network_state_key: (dictionary) Key identifying the model state in the
      checkpoint to restore.
    layer_to_reset: Optional identifying name of the layer to reset when loading
      the checkpoint (useful for resetting the classification layer to use a
      different number of classes for example).
  """

  path: str | None = None
  params_key: str | None = None
  network_state_key: str | None = None
  layer_to_reset: str | None = None


@dataclasses.dataclass(kw_only=True, slots=True)
class ModelConfig:
  """Config for the model.

  Attributes:
    name: Identifying name of the model.
    kwargs: Keyword arguments to construct the model.
    restore: Configuration for restoring the model.
  """
  name: str
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  restore: ModelRestoreConfig = dataclasses.field(
      default_factory=ModelRestoreConfig)


@dataclasses.dataclass(kw_only=True, slots=True)
class ExperimentConfig:
  """Configuration for the experiment.

  Attributes:
    num_updates: Number of updates for the experiment.
    optimizer: Optimizer configuration.
    model: Model configuration.
    training: Training configuration.
    averaging: Averaging configuration.
    evaluation: Evaluation configuration.
    data_train: Training data configuration.
    data_eval: Eval data configuration.
    random_seed: Random seed (automatically changed from the default value).
  """

  num_updates: int
  optimizer: optimizer_config.OptimizerConfig
  model: ModelConfig
  training: experiment_config_py.TrainingConfig
  averaging: experiment_config_py.AveragingConfig
  evaluation: experiment_config_py.EvaluationConfig
  data_train: data.DataLoader
  data_eval: data.DataLoader
  random_seed: int = 0


def build_jaxline_config(
    experiment_config: ExperimentConfig,
) -> ml_collections.ConfigDict:
  """Creates the Jaxline configuration for the experiment."""

  config = jaxline_base_config.get_base_config()

  config.checkpoint_dir = '/tmp/jax_privacy/ckpt_dir'

  # We use same rng for all replicas:
  # (we take care of specializing ourselves the rngs where needed).
  config.random_mode_train = 'same_host_same_device'

  config.random_seed = random.randint(0, 1_000_000)

  # Intervals can be measured in 'steps' or 'secs'.
  config.interval_type = 'steps'
  config.log_train_data_interval = 100
  config.log_tensors_interval = 100
  config.save_checkpoint_interval = 250
  config.eval_specific_checkpoint_dir = ''

  config.experiment_kwargs = ml_collections.ConfigDict()
  config.experiment_kwargs.config = experiment_config

  config.experiment_kwargs.config.random_seed = config.random_seed

  config.experiment_kwargs.config.training.logging.prepend_split_name = True

  # Ensure that random key splitting is configured as expected. The amount of
  # noise injected in DP-SGD will be invalid otherwise.
  assert config.random_mode_train == 'same_host_same_device'

  if config.experiment_kwargs.config.training.dp.auto_tune:
    config = auto_tune.dp_auto_tune_config(config)

  return config
