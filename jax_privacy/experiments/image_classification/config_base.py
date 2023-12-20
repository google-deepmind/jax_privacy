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

from collections.abc import Mapping
import dataclasses
import random

from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification.models import base
from jax_privacy.src.training import auto_tune
from jax_privacy.src.training import averaging as averaging_py
from jax_privacy.src.training import experiment_config as experiment_config_py
from jax_privacy.src.training import optimizer_config
from jaxline import base_config as jaxline_base_config
import ml_collections


@dataclasses.dataclass(kw_only=True, slots=True)
class ExperimentConfig:
  """Configuration for the experiment.

  Attributes:
    optimizer: Optimizer configuration.
    model: Model configuration.
    training: Training configuration.
    label_smoothing: parameter within [0, 1] to smooth the labels. The default
      value of 0 corresponds to no smoothing.
    averaging: Averaging configuration.
    evaluation: Evaluation configuration.
    eval_disparity: Whether to compute disparity at evaluation time.
    data_train: Training data configuration.
    data_eval: Eval data configuration.
    data_eval_additional: Configuration for an (optional) additional evaluation
      dataset.
    random_seed: Random seed (automatically changed from the default value).
  """

  optimizer: optimizer_config.OptimizerConfig
  model: base.ModelConfig
  training: experiment_config_py.TrainingConfig
  label_smoothing: float = 0.0
  averaging: Mapping[str, averaging_py.AveragingConfig] = dataclasses.field(
      default_factory=dict)
  evaluation: experiment_config_py.EvaluationConfig
  eval_disparity: bool = False
  data_train: data.DataLoader
  data_eval: data.DataLoader
  data_eval_additional: data.DataLoader | None = None
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
  config.log_train_data_interval = 10
  config.log_tensors_interval = 10
  config.save_checkpoint_interval = 50
  config.eval_specific_checkpoint_dir = ''

  config.experiment_kwargs = ml_collections.ConfigDict()
  config.experiment_kwargs.config = experiment_config

  config.experiment_kwargs.config.random_seed = config.random_seed

  config.experiment_kwargs.config.training.logging.prepend_split_name = True

  # Ensure that random key splitting is configured as expected. The amount of
  # noise injected in DP-SGD will be invalid otherwise.
  assert config.random_mode_train == 'same_host_same_device'

  if config.experiment_kwargs.config.training.dp.auto_tune_field:
    config.experiment_kwargs.config.training = auto_tune.dp_auto_tune_config(
        config.experiment_kwargs.config.training,
        config.experiment_kwargs.config.data_train.config.num_samples,
    )

  return config
