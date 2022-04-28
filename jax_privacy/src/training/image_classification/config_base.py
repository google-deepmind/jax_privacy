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

"""Base configuration."""

import random
from typing import Callable

from absl import flags
from jax_privacy.src.training import auto_tune
from jaxline import base_config as jaxline_base_config
import ml_collections

FLAGS = flags.FLAGS


MODEL_CKPT = ml_collections.FrozenConfigDict({
    'WRN_40_4_CIFAR100': 'WRN_40_4_CIFAR100.dill',
    'WRN_40_4_IMAGENET32': 'WRN_40_4_IMAGENET32.dill',
    'WRN_28_10_IMAGENET32': 'WRN_28_10_IMAGENET32.dill',
})


def _get_base_config() -> ml_collections.ConfigDict:
  """Return config object for training."""
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

  return config


def wrap_get_config(
    get_config: Callable[[ml_collections.ConfigDict], ml_collections.ConfigDict]
) -> Callable[[], ml_collections.ConfigDict]:
  """Wrap get_config to initialize the config and check validity."""
  def wrapped_get_config():
    base_config = _get_base_config()

    config = get_config(base_config)

    # Ensure that random key splitting is configured as expected. The amount of
    # noise injected in DP-SGD will be invalid otherwise.
    assert config.random_mode_train == 'same_host_same_device'

    if config.experiment_kwargs.config.training.dp.auto_tune:
      config = auto_tune.dp_auto_tune_config(config)

    return config
  return wrapped_get_config
