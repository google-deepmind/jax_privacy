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

"""Tests for auto_tune."""

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.src import accounting
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import auto_tune as auto_tune_py
from jax_privacy.src.training import experiment_config
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
  return experiment_config.TrainingConfig(
      num_updates=100,
      batch_size=experiment_config.BatchSizeTrainConfig(
          total=1024,
          per_device_per_step=8,
      ),
      dp=experiment_config.DPConfig(
          delta=1e-5,
          clipping_norm=None,
          auto_tune_field=auto_tune,
          auto_tune_target_epsilon=2.0,
          noise_multiplier=1.0,
          accountant=accountant_config,
      ),
  )


class AutoTuneTest(parameterized.TestCase):

  def _assert_calibrated(
      self,
      config: experiment_config.TrainingConfig,
      target_eps: float,
  ):
    """Asserts that the config is calibrated wr.t. the target epsilon."""
    eps = accounting.compute_epsilon(
        noise_multipliers=config.dp.noise_multiplier,
        batch_sizes=config.batch_size.total,
        num_steps=config.num_updates,
        num_examples=_NUM_SAMPLES,
        target_delta=config.dp.delta,
        dp_accountant_config=config.dp.accountant,
    )
    np.testing.assert_allclose(eps, target_eps, atol=0.05)

  @parameterized.parameters('rdp', 'pld')
  def test_no_autotune(self, accountant_name):
    config_ref = get_config(auto_tune=None, accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref, num_samples=_NUM_SAMPLES)

    assert config_new.num_updates == config_ref.num_updates
    assert (
        config_new.dp.auto_tune_target_epsilon
        == config_ref.dp.auto_tune_target_epsilon
    )
    assert (
        config_new.dp.noise_multiplier
        == config_ref.dp.noise_multiplier
    )
    assert (
        config_new.batch_size.total
        == config_ref.batch_size.total
    )

  @parameterized.parameters('rdp', 'pld')
  def test_tune_noise_multiplier(self, accountant_name):
    config_ref = get_config(
        auto_tune='noise_multiplier', accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref, num_samples=_NUM_SAMPLES)

    assert (
        config_ref.dp.noise_multiplier != config_new.dp.noise_multiplier
    )
    self._assert_calibrated(config_new, config_ref.dp.auto_tune_target_epsilon)

  @parameterized.parameters('rdp', 'pld')
  def test_tune_num_updates(self, accountant_name):
    config_ref = get_config(
        auto_tune='num_updates', accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref, num_samples=_NUM_SAMPLES)

    assert (
        config_ref.num_updates != config_new.num_updates
    )
    self._assert_calibrated(config_new, config_ref.dp.auto_tune_target_epsilon)

  @parameterized.parameters('rdp', 'pld')
  def test_tune_epsilon(self, accountant_name):
    config_ref = get_config(
        auto_tune='epsilon', accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref, num_samples=_NUM_SAMPLES)

    assert (
        config_ref.dp.auto_tune_target_epsilon
        != config_new.dp.auto_tune_target_epsilon
    )
    self._assert_calibrated(config_new, config_new.dp.auto_tune_target_epsilon)

  @parameterized.parameters('rdp', 'pld')
  def test_tune_batch_size(self, accountant_name):
    config_ref = get_config(
        auto_tune='batch_size', accountant_name=accountant_name)
    config_new = auto_tune_py.dp_auto_tune_config(
        config_ref, num_samples=_NUM_SAMPLES)

    assert (
        config_ref.batch_size.total != config_new.batch_size.total
    )
    self._assert_calibrated(config_new, config_ref.dp.auto_tune_target_epsilon)


if __name__ == '__main__':
  absltest.main()
