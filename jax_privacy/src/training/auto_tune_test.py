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

"""Tests for auto_tune."""
# pylint failing to capture function wrapping
# pylint: disable=no-value-for-parameter

from absl.testing import absltest
from jax_privacy.src import accounting
from jax_privacy.src.training import auto_tune
from jax_privacy.src.training.image_classification import config_base
from jax_privacy.src.training.image_classification import data
import ml_collections
import numpy as np


@config_base.wrap_get_config
def get_config(config):
  """Creates a dummy config for the test."""
  config.experiment_kwargs = ml_collections.ConfigDict(
      {'config': {
          'num_updates': 100,
          'training': {
              'batch_size': {
                  'init_value': 8,
                  'scale_schedule': None,
              },
              'dp': {
                  'target_delta': 1e-5,
                  'auto_tune': None,
                  'noise': {
                      'std_relative': 1.0,
                  }
              },
          },
          'data': {
              'dataset': data.get_dataset('cifar10', 'train', 'valid'),
          }
      },
       }
  )
  return config


class AutoTuneTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = get_config()
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.stop_training_at_epsilon = 1.0

  def _assert_calibrated(self, config, target_eps):
    config_xp = config.experiment_kwargs.config
    eps = accounting.compute_epsilon(
        noise_multipliers=config_xp.training.dp.noise.std_relative,
        batch_sizes=config_xp.training.batch_size.init_value,
        num_steps=config_xp.num_updates,
        num_examples=config_xp.data.dataset.train.num_samples,
        target_delta=config_xp.training.dp.target_delta,
    )
    assert np.isclose(eps, target_eps, rtol=1e-2)

  def test_no_autotune(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = None
    config = auto_tune.dp_auto_tune_config(self.config)

    assert config == self.config

  def test_tune_std_relative(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'std_relative'
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(self.config)

    assert config != get_config()
    self._assert_calibrated(config, target_eps)

  def test_tune_num_updates(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'num_updates'
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(self.config)

    assert config != get_config()
    self._assert_calibrated(config, target_eps)

  def test_tune_epsilon(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'stop_training_at_epsilon'
    config = auto_tune.dp_auto_tune_config(self.config)

    assert config != get_config()

  def test_tune_batch_size(self):
    config_dp = self.config.experiment_kwargs.config.training.dp
    config_dp.auto_tune = 'batch_size'
    target_eps = config_dp.stop_training_at_epsilon
    config = auto_tune.dp_auto_tune_config(self.config)

    assert config != get_config()
    self._assert_calibrated(config, target_eps)


if __name__ == '__main__':
  absltest.main()
