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

"""Test datasets with fake data."""

from absl.testing import absltest
import chex
import jax
from jax_privacy.experiments import image_data

_FAKE_DATA = True
_AUGMULT = 2


def datasets_to_test():
  augmult_config = image_data.AugmultConfig(
      augmult=_AUGMULT,
      random_crop=True,
      random_flip=True,
      random_color=False,
  )
  return (
      image_data.ImageNetLoader(
          config=image_data.ImagenetTrainConfig(
              image_size=(224, 224),
          ),
          augmult_config=augmult_config,
          debug=_FAKE_DATA,
      ),
      image_data.MnistLoader(
          config=image_data.MnistTrainConfig(),
          augmult_config=augmult_config,
          debug=_FAKE_DATA,
      ),
      image_data.Cifar10Loader(
          config=image_data.Cifar10TrainConfig(),
          augmult_config=augmult_config,
          debug=_FAKE_DATA,
      ),
      image_data.Places365Loader(
          config=image_data.Places365TrainConfig(
              image_size=(224, 224),
          ),
          augmult_config=augmult_config,
          debug=_FAKE_DATA,
      ),
  )


class DatasetTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.num_hosts = jax.process_count()
    self.num_devices = jax.local_device_count()
    self.local_batch_size = 4

  def test_dataset(self):
    for dataset in datasets_to_test():
      images_train_shape = (
          self.num_devices,
          self.local_batch_size,
          _AUGMULT,
          *dataset.config.image_size,
      )
      labels_train_shape = (
          self.num_devices,
          self.local_batch_size,
          _AUGMULT,
          dataset.config.num_classes,
      )

      data_train = dataset.load_dataset(
          batch_dims=(self.num_devices, self.local_batch_size),
          shard_data=True,
          is_training=True,
      )
      batch_train = next(iter(data_train))

      # Assert shape, except on the channel dimension, which is unknown to
      # images_train_shape.
      chex.assert_tree_shape_prefix(batch_train.image, images_train_shape)
      chex.assert_shape(batch_train.label, labels_train_shape)

      images_eval_shape = (
          self.num_hosts,
          self.num_devices,
          self.local_batch_size,
          *dataset.config.image_size,
      )
      labels_eval_shape = (
          self.num_hosts,
          self.num_devices,
          self.local_batch_size,
          dataset.config.num_classes,
      )

      data_eval = dataset.load_dataset(
          batch_dims=(self.num_hosts, self.num_devices, self.local_batch_size),
          shard_data=False,
          is_training=False,
      )
      batch_eval = next(iter(data_eval))

      # Assert shape, except on the channel dimension, which is unknown to
      # images_eval_shape.
      chex.assert_tree_shape_prefix(batch_eval.image, images_eval_shape)
      chex.assert_shape(batch_eval.label, labels_eval_shape)


if __name__ == '__main__':
  absltest.main()
