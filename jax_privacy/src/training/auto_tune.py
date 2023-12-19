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

"""Auto-tune DP parameters of config so that they fit the privacy budget."""

import dataclasses

from absl import logging
from jax_privacy.src import accounting as dp_accounting
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import experiment_config


def dp_auto_tune(
    *,
    auto_tune: typing.AutoTuneField,
    num_examples: int,
    dp_epsilon: float,
    dp_delta: float,
    noise_multiplier: float,
    batch_sizes: int,
    num_updates: int,
    dp_accountant_config: dp_accounting.DpAccountantConfig,
) -> tuple[float, int, float, int]:
  """Auto-tune DP parameters so that we can obtain the desired DP guarantees.

  Args:
    auto_tune: which hyper-parameter to adapt.
    num_examples: number of examples in the training set.
    dp_epsilon: epsilon-value of DP guarantee.
    dp_delta: delta-value of DP guarantee.
    noise_multiplier: standard deviation of the noise (relative to the
      clipping-norm).
    batch_sizes: batch-size used during training.
    num_updates: number of updates to be performed.
    dp_accountant_config: Configuration for the DP accountant to use.

  Returns:
    Potentially updated values for dp_epsilon, num_updates, noise_multiplier,
    and batch_sizes.
  """

  if not auto_tune:
    pass
  elif auto_tune == 'epsilon':
    dp_epsilon: float = dp_accounting.compute_epsilon(
        noise_multipliers=noise_multiplier,
        batch_sizes=batch_sizes,
        num_steps=num_updates,
        num_examples=num_examples,
        target_delta=dp_delta,
        dp_accountant_config=dp_accountant_config,
    )
  elif auto_tune == 'num_updates':
    num_updates: int = dp_accounting.calibrate_steps(
        target_epsilon=dp_epsilon,
        noise_multipliers=noise_multiplier,
        batch_sizes=batch_sizes,
        num_examples=num_examples,
        target_delta=dp_delta,
        dp_accountant_config=dp_accountant_config,
    )
  elif auto_tune == 'noise_multiplier':
    noise_multiplier: float = dp_accounting.calibrate_noise_multiplier(
        target_epsilon=dp_epsilon,
        num_steps=num_updates,
        batch_sizes=batch_sizes,
        num_examples=num_examples,
        target_delta=dp_delta,
        dp_accountant_config=dp_accountant_config,
    )
  elif auto_tune == 'batch_size':
    batch_sizes: int = dp_accounting.calibrate_batch_size(
        target_epsilon=dp_epsilon,
        noise_multipliers=noise_multiplier,
        num_steps=num_updates,
        num_examples=num_examples,
        target_delta=dp_delta,
        dp_accountant_config=dp_accountant_config,
    )
  else:
    raise ValueError(f'Unsupported auto-tuning option: {auto_tune}.')

  return dp_epsilon, num_updates, noise_multiplier, batch_sizes


def dp_auto_tune_config(
    training_config: experiment_config.TrainingConfig,
    num_samples: int,
) -> experiment_config.TrainingConfig:
  """Apply DP auto-tuning to the config (modified in-place)."""
  if training_config.batch_size.scale_schedule is not None:
    raise ValueError('Batch-size schedules are not supported.')
  dp_accountant_config = training_config.dp.accountant
  if isinstance(dp_accountant_config, dp_accounting.PldAccountantConfig):
    logging.warning(
        'Auto tuning with PLD accountant can be slow. Be patient...'
    )

  epsilon, num_updates, noise_multiplier, batch_size = dp_auto_tune(
      batch_sizes=training_config.batch_size.total,
      noise_multiplier=training_config.dp.noise_multiplier,
      dp_epsilon=training_config.dp.auto_tune_target_epsilon,
      num_updates=training_config.num_updates,
      auto_tune=training_config.dp.auto_tune_field,
      num_examples=num_samples,
      dp_delta=training_config.dp.delta,
      dp_accountant_config=dp_accountant_config,
  )

  training_config = dataclasses.replace(
      training_config,
      num_updates=num_updates,
      batch_size=dataclasses.replace(
          training_config.batch_size,
          total=batch_size,
      ),
      dp=dataclasses.replace(
          training_config.dp,
          auto_tune_target_epsilon=epsilon,
          noise_multiplier=noise_multiplier,
      )
  )

  return training_config
