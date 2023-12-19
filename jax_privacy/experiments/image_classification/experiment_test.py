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

"""Integration test."""

import functools

from absl.testing import absltest
import chex
import jax
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import experiment as experiment_py
from jax_privacy.experiments.image_classification.models import models
from jax_privacy.src.training import averaging
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config


def get_config() -> config_base.ExperimentConfig:
  """Creates a dummy config for the test."""
  config = config_base.ExperimentConfig(
      optimizer=optimizer_config.OptimizerConfig(
          name='sgd',
          lr=optimizer_config.constant_lr_config(4.0),
      ),
      training=experiment_config.TrainingConfig(
          num_updates=3,
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=8,
              per_device_per_step=4,
          ),
          dp=experiment_config.DPConfig.deactivated(),
      ),
      model=models.WideResNetConfig(
          depth=4,
          width=1,
          groups=2,
      ),
      data_train=data.Cifar10Loader(
          config=data.Cifar10TrainConfig(),
          debug=True,
      ),
      data_eval=data.Cifar10Loader(
          config=data.Cifar10ValidConfig(),
          debug=True,
      ),
      evaluation=experiment_config.EvaluationConfig(
          batch_size=1,
      ),
      averaging={
          'ema': averaging.ExponentialMovingAveragingConfig(decay=0.9),
          'polyak': averaging.PolyakAveragingConfig(),
      },
  )

  return config


class ExperimentTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    config = get_config()
    self._experiment = experiment_py.ImageClassificationExperiment(config)
    self._rng_init = jax.random.PRNGKey(28634)

  def testTrain(self):
    state, step_on_host, data_train = self._experiment.initialize(
        self._rng_init,
    )
    self._experiment.update(
        state, functools.partial(next, data_train), step_on_host)

  def testEval(self):
    state, step_on_host, unused_data_train = self._experiment.initialize(
        self._rng_init,
    )
    self._experiment.evaluate(state=state, step_on_host=step_on_host)


if __name__ == '__main__':
  absltest.main()
