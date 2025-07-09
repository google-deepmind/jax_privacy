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

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import accounting
from jax_privacy.dp_sgd import typing
from jax_privacy.training import algorithm_config
from jax_privacy.training import auto_tune as auto_tune_py
from jax_privacy.training import experiment_config
import numpy as np


_NUM_SAMPLES = 50_000


def get_config(
    auto_tune: typing.AutoTuneField,
    accountant_name: str,
) -> experiment_config.TrainingConfig:
  """Returns a TrainingConfig for testing."""
  if accountant_name == 'pld':
    accountant_config = accounting.PldAccountantConfig(
        value_discretization_interval=1e-2,
    )
  elif accountant_name == 'rdp':
    accountant_config = accounting.RdpAccountantConfig()
  else:
    raise ValueError(f'Unknown accountant: {accountant_name}.')
  algorithm = algorithm_config.DpsgdConfig(noise_multiplier=1.0)
  return experiment_config.TrainingConfig(
      num_updates=100,
      batch_size=experiment_config.BatchSizeTrainConfig(
          total=1024,
          per_device_per_step=8,
      ),
      dp=experiment_config.DpConfig(
          delta=1e-5,
          clipping_norm=1.0,
          auto_tune_field=auto_tune,
          auto_tune_target_epsilon=2.0,
          algorithm=algorithm,
          dp_accountant=accountant_config,
      ),
  )


class AutoTuneTest(parameterized.TestCase):

  def _assert_calibrated(
      self,
      config: experiment_config.TrainingConfig,
      target_eps: float,
      *,
      num_examples_per_user: int,
      cycle_length: int,
      truncated_batch_size: int,
  ):
    """Asserts that the config is calibrated w.r.t. the target epsilon."""
    dp_params = accounting.DpParams(
        noise_multipliers=config.dp.algorithm.noise_multiplier,
        batch_size=config.batch_size.total,
        num_samples=_NUM_SAMPLES,
        delta=config.dp.delta,
        examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
    if num_examples_per_user is None:
      accountant = accounting.DpsgdTrainingAccountant(
          dp_accountant_config=config.dp.dp_accountant
      )
    else:
      accountant = accounting.DpsgdTrainingUserLevelAccountant(
          dp_accountant_config=config.dp.dp_accountant
      )
    eps = accountant.compute_epsilon(
        num_updates=config.num_updates, dp_params=dp_params
    )
    np.testing.assert_allclose(eps, target_eps, atol=0.05)

  @parameterized.parameters(
      ('rdp', None),
      ('rdp', 1),
      ('pld', None),
  )
  def test_no_autotune(self, accountant_name, num_examples_per_user):
    config_ref = get_config(auto_tune=None, accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref,
        num_samples=_NUM_SAMPLES,
        num_examples_per_user=num_examples_per_user,
    )

    assert config_new.num_updates == config_ref.num_updates
    assert (
        config_new.dp.auto_tune_target_epsilon
        == config_ref.dp.auto_tune_target_epsilon
    )
    assert (
        config_new.dp.algorithm.noise_multiplier
        == config_ref.dp.algorithm.noise_multiplier
    )
    assert config_new.batch_size.total == config_ref.batch_size.total

  # User level accounting and truncation are supported only for PLD.
  @parameterized.parameters(
      ('rdp', None, None, None),
      ('rdp', None, 2, None),
      ('pld', None, None, None),
      ('pld', None, 2, None),
      ('pld', 1, None, None),
      ('pld', None, None, 1056),
  )
  def test_tune_noise_multiplier(
      self,
      accountant_name,
      num_examples_per_user,
      cycle_length,
      truncated_batch_size,
  ):
    config_ref = get_config(
        auto_tune='noise_multiplier',
        accountant_name=accountant_name,
    )
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref,
        num_samples=_NUM_SAMPLES,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

    assert (
        config_ref.dp.algorithm.noise_multiplier
        != config_new.dp.algorithm.noise_multiplier
    )
    self._assert_calibrated(
        config_new,
        config_ref.dp.auto_tune_target_epsilon,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

  # User level accounting and truncation are supported only for PLD.
  @parameterized.parameters(
      ('rdp', None, None, None),
      ('rdp', None, 2, None),
      ('pld', None, None, None),
      ('pld', None, 2, None),
      ('pld', 1, None, None),
      ('pld', None, None, 1112),
  )
  def test_tune_num_updates(
      self,
      accountant_name,
      num_examples_per_user,
      cycle_length,
      truncated_batch_size,
  ):
    config_ref = get_config(
        auto_tune='num_updates',
        accountant_name=accountant_name,
    )
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref,
        num_samples=_NUM_SAMPLES,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

    assert config_ref.num_updates != config_new.num_updates
    self._assert_calibrated(
        config_new,
        config_ref.dp.auto_tune_target_epsilon,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

  # User level accounting and truncation are supported only for PLD.
  @parameterized.parameters(
      ('rdp', None, None, None),
      ('rdp', None, 2, None),
      ('pld', None, None, None),
      ('pld', None, 2, None),
      ('pld', 1, None, None),
      ('pld', 2, None, None),
      ('pld', None, None, 1056),
  )
  def test_tune_epsilon(
      self,
      accountant_name,
      num_examples_per_user,
      cycle_length,
      truncated_batch_size,
  ):
    config_ref = get_config(
        auto_tune='epsilon',
        accountant_name=accountant_name,
    )
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref,
        num_samples=_NUM_SAMPLES,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

    assert (
        config_ref.dp.auto_tune_target_epsilon
        != config_new.dp.auto_tune_target_epsilon
    )
    self._assert_calibrated(
        config_new,
        config_new.dp.auto_tune_target_epsilon,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

  # user level accounting is supported only for PLD.
  @parameterized.parameters(
      ('rdp', None, None, None),
      ('rdp', None, 2, None),
      ('pld', None, None, None),
      ('pld', None, 2, None),
      ('pld', 1, None, None),
      ('pld', None, None, 1056),
  )
  def test_tune_batch_size(
      self,
      accountant_name,
      num_examples_per_user,
      cycle_length,
      truncated_batch_size,
  ):
    config_ref = get_config(
        auto_tune='batch_size',
        accountant_name=accountant_name,
    )
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref,
        num_samples=_NUM_SAMPLES,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )

    assert config_ref.batch_size.total != config_new.batch_size.total
    self._assert_calibrated(
        config_new,
        config_ref.dp.auto_tune_target_epsilon,
        num_examples_per_user=num_examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )


if __name__ == '__main__':
  absltest.main()
