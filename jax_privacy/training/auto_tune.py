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

"""Auto-tune DP parameters of config so that they fit the privacy budget."""

import dataclasses

from absl import logging
from jax_privacy import accounting
from jax_privacy.training import algorithm_config
from jax_privacy.training import experiment_config


def dp_auto_tune(
    *,
    batch_sizes: int,
    num_samples: int,
    num_updates: int,
    dp_config: experiment_config.DpConfig,
    num_examples_per_user: int | None = None,
    cycle_length: int | None = None,
    truncated_batch_size: int | None = None,
) -> tuple[float | None, int, float, int]:
  """Auto-tune DP parameters so that we can obtain the desired DP guarantees.

  Args:
    batch_sizes: batch-size used during training.
    num_samples: number of examples in the training set.
    num_updates: number of updates to be performed.
    dp_config: Configuration for DP.
    num_examples_per_user: Number of examples per user.
    cycle_length: If using cyclic Poisson sampling with BandMF, the length of
      the cycle.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to.

  Returns:
    Potentially updated values for dp_epsilon, num_updates, noise_multiplier,
    and batch_sizes.
  """
  auto_tune = dp_config.auto_tune_field
  # Guaranteed to be float from DpConfig. If not, this will error.
  noise_multiplier = float(dp_config.algorithm.noise_multiplier)
  epsilon = dp_config.auto_tune_target_epsilon

  if not auto_tune:
    return epsilon, num_updates, noise_multiplier, batch_sizes

  if isinstance(dp_config.dp_accountant, accounting.PldAccountantConfig):
    logging.warning(
        'Auto tuning with PLD accountant can be slow. Be patient...'
    )

  accountant, dp_params = dp_config.make_accountant(
      num_samples=num_samples,
      batch_size=batch_sizes,
      examples_per_user=num_examples_per_user,
      cycle_length=cycle_length,
      truncated_batch_size=truncated_batch_size,
  )
  if auto_tune == 'epsilon':
    epsilon: float = accountant.compute_epsilon(num_updates, dp_params)
  elif auto_tune == 'num_updates':
    num_updates: int = accounting.calibrate_num_updates(
        target_epsilon=epsilon,
        accountant=accountant,
        noise_multipliers=noise_multiplier,
        batch_sizes=batch_sizes,
        num_samples=num_samples,
        target_delta=dp_config.delta,
        examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
  elif auto_tune == 'noise_multiplier':
    noise_multiplier: float = accounting.calibrate_noise_multiplier(
        target_epsilon=epsilon,
        accountant=accountant,
        batch_sizes=batch_sizes,
        num_updates=num_updates,
        num_samples=num_samples,
        target_delta=dp_config.delta,
        examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
  elif auto_tune == 'batch_size':
    batch_sizes: int = accounting.calibrate_batch_size(
        target_epsilon=epsilon,
        accountant=accountant,
        noise_multipliers=noise_multiplier,
        num_updates=num_updates,
        num_samples=num_samples,
        target_delta=dp_config.delta,
        examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
  else:
    raise ValueError(f'Unsupported auto-tuning option: {auto_tune}.')

  return epsilon, num_updates, noise_multiplier, batch_sizes


def dp_auto_tune_config(
    training_config: experiment_config.TrainingConfig,
    num_samples: int,
    num_examples_per_user: int | None = None,
    cycle_length: int | None = None,
    truncated_batch_size: int | None = None,
) -> experiment_config.TrainingConfig:
  """Apply DP auto-tuning to the config (modified in-place)."""
  if training_config.batch_size.scale_schedule is not None:
    raise ValueError('Batch-size schedules are not supported.')

  epsilon, num_updates, noise_multiplier, batch_size = dp_auto_tune(
      batch_sizes=training_config.batch_size.total,
      num_updates=training_config.num_updates,
      num_samples=num_samples,
      dp_config=training_config.dp,
      num_examples_per_user=num_examples_per_user,
      cycle_length=cycle_length,
      truncated_batch_size=truncated_batch_size,
  )

  algorithm = algorithm_config.DpsgdConfig(noise_multiplier=noise_multiplier)
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
          algorithm=algorithm,
      ),
  )

  return training_config
