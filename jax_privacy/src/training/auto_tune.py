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

"""Auto-tune DP parameters of config so that they fit the privacy budget."""
from typing import Tuple

from jax_privacy.src import accounting as dp_accounting
import ml_collections


def dp_auto_tune(
    *,
    auto_tune: str,
    num_examples: int,
    dp_epsilon: float,
    dp_delta: float,
    std_relative: float,
    batch_sizes: int,
    num_updates: int,
) -> Tuple[float, int, float, int]:
  """Auto-tune DP parameters so that we can obtain the desired DP guarantees.

  Args:
    auto_tune: which hyper-parameter to adapt.
    num_examples: number of examples in the training set.
    dp_epsilon: epsilon-value of DP guarantee.
    dp_delta: delta-value of DP guarantee.
    std_relative: standard deviation relative to the clipping-norm (aka noise
      multiplier).
    batch_sizes: batch-size used during training.
    num_updates: number of updates to be performed.

  Returns:
    Potentially updated values for dp_epsilon, num_updates, std_relative, and
    batch_sizes.
  """

  if not auto_tune:
    pass
  elif auto_tune == 'stop_training_at_epsilon':
    dp_epsilon: float = dp_accounting.compute_epsilon(
        noise_multipliers=std_relative,
        batch_sizes=batch_sizes,
        num_steps=num_updates,
        num_examples=num_examples,
        target_delta=dp_delta,
    )
  elif auto_tune == 'num_updates':
    num_updates: int = dp_accounting.calibrate_steps(
        target_epsilon=dp_epsilon,
        noise_multipliers=std_relative,
        batch_sizes=batch_sizes,
        num_examples=num_examples,
        target_delta=dp_delta,
    )
  elif auto_tune == 'std_relative':
    std_relative: float = dp_accounting.calibrate_noise_multiplier(
        target_epsilon=dp_epsilon,
        num_steps=num_updates,
        batch_sizes=batch_sizes,
        num_examples=num_examples,
        target_delta=dp_delta,
    )
  elif auto_tune == 'batch_size':
    batch_sizes: int = dp_accounting.calibrate_batch_size(
        target_epsilon=dp_epsilon,
        noise_multipliers=std_relative,
        num_steps=num_updates,
        num_examples=num_examples,
        target_delta=dp_delta,
    )
  else:
    raise ValueError(f'Unsupported auto-tuning option: {auto_tune}.')

  return dp_epsilon, num_updates, std_relative, batch_sizes


def dp_auto_tune_config(
    config: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Apply DP auto-tuning to the config (modified in-place)."""
  config_xp = config.experiment_kwargs.config
  if config_xp.training.batch_size.scale_schedule is not None:
    raise ValueError('Batch-size schedules are not supported.')

  epsilon, num_updates, std_relative, batch_size = dp_auto_tune(
      batch_sizes=config_xp.training.batch_size.init_value,
      std_relative=config_xp.training.dp.noise.std_relative,
      dp_epsilon=config_xp.training.dp.stop_training_at_epsilon,
      num_updates=config_xp.num_updates,
      auto_tune=config_xp.training.dp.auto_tune,
      num_examples=config_xp.data.dataset.train.num_samples,
      dp_delta=config_xp.training.dp.target_delta,
  )

  config_xp.num_updates = num_updates
  config_xp.training.dp.stop_training_at_epsilon = epsilon
  config_xp.training.dp.noise.std_relative = std_relative
  config_xp.training.batch_size.init_value = batch_size
  return config
