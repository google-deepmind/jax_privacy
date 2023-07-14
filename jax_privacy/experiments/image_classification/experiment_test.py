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

from unittest import mock

from absl.testing import absltest
import chex
import jax
from jax import random
import jax.numpy as jnp
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import experiment
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config
from jaxline import train
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Creates a dummy config for the test."""
  config = config_base.ExperimentConfig(
      num_updates=3,
      optimizer=optimizer_config.OptimizerConfig(
          name='sgd',
          lr=optimizer_config.constant_lr_config(4.0),
      ),
      training=experiment_config.TrainingConfig(
          batch_size=experiment_config.BatchSizeTrainConfig(
              total=8,
              per_device_per_step=4,
          ),
          dp=experiment_config.NoDPConfig(),
      ),
      model=config_base.ModelConfig(
          name='wideresnet',
          kwargs={
              'depth': 4,
              'width': 1,
              'groups': 2,
          },
          restore=config_base.ModelRestoreConfig(
              path=None,
          ),
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
      averaging=experiment_config.AveragingConfig(
          ema_enabled=True,
          ema_coefficient=0.9999,
          ema_start_step=0,
          polyak_enabled=True,
          polyak_start_step=0,
      ),
  )

  return config_base.build_jaxline_config(config)


class ExperimentTest(chex.TestCase):

  def testTrain(self):
    cfg = get_config()
    train.train(experiment.Experiment, cfg,
                checkpointer=mock.Mock(), writer=mock.Mock())

  def testEval(self):
    cfg = get_config()
    rng = random.PRNGKey(cfg.random_seed)
    exp = experiment.Experiment('eval', init_rng=rng, **cfg.experiment_kwargs)
    rng = jnp.broadcast_to(rng, (jax.local_device_count(),) + rng.shape)
    global_step = jnp.broadcast_to(0, [jax.local_device_count()])
    # Simulate a checkpoint restore.
    exp.step(global_step=global_step, rng=rng, writer=None)
    exp.evaluate(global_step=global_step, rng=rng, writer=None)


if __name__ == '__main__':
  absltest.main()
