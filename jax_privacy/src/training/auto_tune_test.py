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

import copy

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.src import accounting
from jax_privacy.src.training import auto_tune
from jax_privacy.src.training import experiment_config
import ml_collections
import numpy as np


def get_config() -> ml_collections.ConfigDict:
  """Creates a dummy config for the test."""
  config = config_base.ExperimentConfig(  # pytype: disable=wrong-arg-types
      num_updates=100,
      optimizer=None,
      model=None,
      training=experiment_config.TrainingConfig(
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=1024,
              per_device_per_step=8,
          ),
          dp=experiment_config.DPConfig(
              delta=1e-5,
              clipping_norm=None,
              auto_tune=None,
              stop_training_at_epsilon=2.0,
              noise_multiplier=1.0,
              accountant=accounting.RdpAccountantConfig(),
          ),
      ),
      averaging=None,
      random_seed=None,
      data_train=image_data.Cifar10Loader(
          config=image_data.Cifar10TrainValidConfig(
              preprocess_name='standardise',
          ),
      ),
      data_eval=image_data.Cifar10Loader(
          config=image_data.Cifar10TestConfig(
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


class AutoTuneTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.config = get_config()

    self._accountant = {
        'pld': accounting.PldAccountantConfig(
            value_discretization_interval=1e-2,
        ),
        'rdp': accounting.RdpAccountantConfig(),
    }

  def _assert_calibrated(self, config, target_eps):
    """Asserts that the config is calibrated wr.t. the target epsilon."""
    config_xp = config.experiment_kwargs.config
    eps = accounting.compute_epsilon(
        noise_multipliers=config_xp.training.dp.noise_multiplier,
        batch_sizes=config_xp.training.batch_size.total,
        num_steps=config_xp.num_updates,
        num_examples=config_xp.data_train.config.num_samples,
        target_delta=config_xp.training.dp.delta,
        dp_accountant_config=config_xp.training.dp.accountant,
    )
    np.testing.assert_allclose(eps, target_eps, atol=0.05)

  def test_no_autotune(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = None
    config = auto_tune.dp_auto_tune_config(copy.deepcopy(self.config))

    config_xp_after = config.experiment_kwargs.config
    config_xp_before = self.config.experiment_kwargs.config

    assert config_xp_after.num_updates == config_xp_before.num_updates
    assert (
        config_xp_after.training.dp.stop_training_at_epsilon
        == config_xp_before.training.dp.stop_training_at_epsilon
    )
    assert (
        config_xp_after.training.dp.noise_multiplier
        == config_xp_before.training.dp.noise_multiplier
    )
    assert (
        config_xp_after.training.batch_size.total
        == config_xp_before.training.batch_size.total
    )

  @parameterized.parameters('rdp', 'pld')
  def test_tune_noise_multiplier(self, accountant_name):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'noise_multiplier'
    config_dp.accountant = self._accountant[accountant_name]
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(copy.deepcopy(self.config))

    assert (
        config.experiment_kwargs.config != self.config.experiment_kwargs.config
    )
    self._assert_calibrated(config, target_eps)

  @parameterized.parameters('rdp', 'pld')
  def test_tune_num_updates(self, accountant_name):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'num_updates'
    config_dp.accountant = self._accountant[accountant_name]
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(copy.deepcopy(self.config))

    assert (
        config.experiment_kwargs.config != self.config.experiment_kwargs.config
    )
    self._assert_calibrated(config, target_eps)

  @parameterized.parameters('rdp', 'pld')
  def test_tune_epsilon(self, accountant_name):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'stop_training_at_epsilon'
    config_dp.accountant = self._accountant[accountant_name]
    config = auto_tune.dp_auto_tune_config(copy.deepcopy(self.config))

    assert (
        config.experiment_kwargs.config != self.config.experiment_kwargs.config
    )
    self._assert_calibrated(
        config,
        config.experiment_kwargs.config.training.dp.stop_training_at_epsilon,
    )

  @parameterized.parameters('rdp', 'pld')
  def test_tune_batch_size(self, accountant_name):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'batch_size'
    config_dp.accountant = self._accountant[accountant_name]
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(copy.deepcopy(self.config))

    assert (
        config.experiment_kwargs.config != self.config.experiment_kwargs.config
    )
    self._assert_calibrated(config, target_eps)


if __name__ == '__main__':
  absltest.main()
