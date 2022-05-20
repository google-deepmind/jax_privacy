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

"""Integration test."""
# pylint failing to capture function wrapping
# pylint: disable=no-value-for-parameter

from unittest import mock

from absl.testing import absltest
import chex
import jax
from jax import random
import jax.numpy as jnp

from jax_privacy.src.training.image_classification import config_base
from jax_privacy.src.training.image_classification import data
from jax_privacy.src.training.image_classification import experiment
from jax_privacy.src.training.image_classification.data import mnist_cifar_svhn

from jaxline import train
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds


@config_base.wrap_get_config
def get_config(config):
  """Creates a dummy config for the test."""
  config.experiment_kwargs = ml_collections.ConfigDict(
      {'config': {
          'num_updates': 100,
          'optimizer': {
              'name': 'sgd',
              'lr': {
                  'init_value': 4.0,
                  'decay_schedule_name': None,
                  'decay_schedule_kwargs': None,
                  'relative_schedule_kwargs': None,
              },
              'kwargs': {},
          },
          'training': {
              'batch_size': {
                  'init_value': 8,
                  'per_device_per_step': 4,
                  'scale_schedule': None,
              },
              'weight_decay': 0.0,
              'train_only_layer': None,
              'dp': {
                  'stop_training_at_epsilon': None,
                  'rescale_to_unit_norm': True,
                  'clipping_norm': 0.1,
                  'target_delta': 1e-5,
                  'auto_tune': None,
                  'noise': {
                      'std_relative': 1.0,
                  }
              },
              'logging': {
                  'grad_clipping': True,
                  'grad_alignment': False,
                  'snr_global': True,  # signal-to-noise ratio across layers
                  'snr_per_layer': False,  # signal-to-noise ratio per layer
              },
          },
          'model': {
              'model_type': 'wideresnet',
              'model_kwargs': {
                  'depth': 10,
                  'width': 1,
                  'groups': 2,
              },
              'restore': {
                  'path': None,
              },
          },
          'data': {
              'dataset': data.get_dataset('cifar10', 'train', 'valid'),
              'random_flip': True,
              'random_crop': True,
              'augmult': 16,  # implements arxiv.org/abs/2105.13343
          },
          'averaging': {
              'ema': {
                  'coefficient': 0.9999,
                  'start_step': 0,
              },
              'polyak': {
                  'start_step': 0,
                  },
              },
          'evaluation': {
              'batch_size': 100,
          },
      },
      }
  )
  return config


class HermeticExperiment(experiment.Experiment):
  """Experiment but with a dummy dataset to make the test hermetic."""

  def _preprocess_batch(self, x, y, is_training):
    """Pre-process the mini-batch like CIFAR data."""
    x, y = mnist_cifar_svhn.preprocess_batch(
        x,
        y,
        is_training=is_training,
        augmult=self.config.data.augmult,
        random_flip=self.config.data.random_flip,
        random_crop=self.config.data.random_crop,
        dataset=self.config.data.dataset,
        image_resize=None,
    )
    return {'images': x, 'labels': y}

  def _build_train_input(self):
    """See base class."""

    return tfds.as_numpy(
        tf.data.Dataset.from_tensors((tf.ones((32, 32, 3)), tf.constant([0])))
        .repeat()
        .map(lambda x, y: self._preprocess_batch(x, y, is_training=True))
        .batch(self.config.training.batch_size.per_device_per_step,
               drop_remainder=True)
        .batch(jax.local_device_count(), drop_remainder=True)
    )

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""

    return tfds.as_numpy(
        tf.data.Dataset.from_tensors((tf.ones((32, 32, 3)), tf.constant([0])))
        .repeat(10)
        .map(lambda x, y: self._preprocess_batch(x, y, is_training=False))
        .batch(2)
    )


class ExperimentTest(chex.TestCase):

  def testTrain(self):
    cfg = get_config()
    train.train(HermeticExperiment, cfg,
                checkpointer=mock.Mock(), writer=mock.Mock())

  def testEval(self):
    cfg = get_config()
    rng = random.PRNGKey(cfg.random_seed)
    exp = HermeticExperiment('eval', init_rng=rng, **cfg.experiment_kwargs)
    rng = jnp.broadcast_to(rng, (jax.local_device_count(),) + rng.shape)
    global_step = jnp.broadcast_to(0, [jax.local_device_count()])
    # Simulate a checkpoint restore.
    exp.step(global_step=global_step, rng=rng, writer=None)
    exp.evaluate(global_step=global_step, rng=rng, writer=None)


if __name__ == '__main__':
  absltest.main()
