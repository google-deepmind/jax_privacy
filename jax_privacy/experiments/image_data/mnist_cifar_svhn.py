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

"""Data loading functions for MNIST / CIFAR / SVHN."""

import dataclasses
import functools

from jax_privacy.experiments.image_data import base
from jax_privacy.experiments.image_data import loader
import tensorflow as tf
import tensorflow_datasets as tfds


MEAN_RGB = (0.49139968, 0.48215841, 0.44653091)
STDDEV_RGB = (0.24703223, 0.24348513, 0.26158784)


@dataclasses.dataclass(kw_only=True, slots=True)
class _DatasetConfig(base.ImageDatasetConfig):
  """Builds the input pipeline for MNIST / SVHN / CIFAR.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train[:50000]".
    name: Unique identifying name for the dataset.
    image_size: Image resolution to use.
    using_large_images: Whether to decode images as large images.
    preprocess_name: Name for the preprocessing function.
  """

  num_samples: int
  num_classes: int
  name: str
  split_content: str
  image_size: tuple[int, int]
  preprocess_name: str | None = None

  def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
    if self.name in ('cifar10', 'cifar100', 'svhn_cropped', 'mnist'):
      if self.name == 'mnist':
        return base.center_image(image)
      elif self.preprocess_name:
        if self.preprocess_name == 'standardise':
          return base.standardize_image_per_channel(
              image,
              mean_per_channel=MEAN_RGB,
              stddev_per_channel=STDDEV_RGB,
          )
        elif self.preprocess_name == 'center':
          return base.center_image(image)
        elif self.preprocess_name == 'none':
          return image
        else:
          raise ValueError(
              'Unexpected preprocessing function: '
              f'{self.preprocess_name}.')
      else:
        return base.standardize_image_per_channel(
            image,
            mean_per_channel=MEAN_RGB,
            stddev_per_channel=STDDEV_RGB,
        )
    else:
      raise ValueError(f'Invalid dataset {self.name}.')

  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input label."""
    return tf.one_hot(label, depth=self.num_classes)


@dataclasses.dataclass(kw_only=True, slots=True)
class _DataLoader(loader.DataLoader):
  """Data loader for MNIST / CIFAR / SVHN."""

  config: _DatasetConfig

  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    ds = tfds.load(
        name=self.config.name,
        split=self.config.split_content,
        shuffle_files=shuffle_files,
    )
    return ds.map(base.DataInputs.from_dict)


MnistLoader = Cifar10Loader = Cifar100Loader = SvhnLoader = _DataLoader

Cifar10TrainConfig = functools.partial(
    _DatasetConfig,
    name='cifar10',
    image_size=(32, 32),
    num_classes=10,
    split_content='train[:45000]',
    num_samples=45_000,
)
Cifar10TrainValidConfig = functools.partial(
    _DatasetConfig,
    name='cifar10',
    image_size=(32, 32),
    num_classes=10,
    split_content='train',
    num_samples=50_000,
)
Cifar10ValidConfig = functools.partial(
    _DatasetConfig,
    name='cifar10',
    image_size=(32, 32),
    num_classes=10,
    split_content='train[45000:]',
    num_samples=5_000,
)
Cifar10TestConfig = functools.partial(
    _DatasetConfig,
    name='cifar10',
    image_size=(32, 32),
    num_classes=10,
    split_content='test',
    num_samples=5_000,
)
Cifar100TrainConfig = functools.partial(
    _DatasetConfig,
    name='cifar100',
    image_size=(32, 32),
    num_classes=100,
    split_content='train[:45000]',
    num_samples=45_000,
)
Cifar100TrainValidConfig = functools.partial(
    _DatasetConfig,
    name='cifar100',
    image_size=(32, 32),
    num_classes=100,
    split_content='train',
    num_samples=50_000,
)
Cifar100ValidConfig = functools.partial(
    _DatasetConfig,
    name='cifar100',
    image_size=(32, 32),
    num_classes=100,
    split_content='train[45000:]',
    num_samples=5_000,
)
Cifar100TestConfig = functools.partial(
    _DatasetConfig,
    name='cifar100',
    image_size=(32, 32),
    num_classes=100,
    split_content='test',
    num_samples=5_000,
)
SvhnTrainConfig = functools.partial(
    _DatasetConfig,
    name='svhn_cropped',
    image_size=(28, 28),
    num_classes=10,
    num_samples=68_257,
    split_content='train[:68257]',
)
SvhnValidConfig = functools.partial(
    _DatasetConfig,
    name='svhn_cropped',
    image_size=(28, 28),
    num_classes=10,
    num_samples=5_000,
    split_content='train[68257:]',
)
SvhnTrainValidConfig = functools.partial(
    _DatasetConfig,
    name='svhn_cropped',
    image_size=(28, 28),
    num_classes=10,
    num_samples=73_257,
    split_content='train',
)
SvhnTestConfig = functools.partial(
    _DatasetConfig,
    name='svhn_cropped',
    image_size=(28, 28),
    num_classes=10,
    num_samples=26_032,
    split_content='test',
)
MnistTrainConfig = functools.partial(
    _DatasetConfig,
    name='mnist',
    image_size=(28, 28),
    num_classes=10,
    num_samples=50_000,
    split_content='train[:50_000]',
)
MnistValidConfig = functools.partial(
    _DatasetConfig,
    name='mnist',
    image_size=(28, 28),
    num_classes=10,
    num_samples=10_000,
    split_content='train[50_000:]',
)
MnistTrainValidConfig = functools.partial(
    _DatasetConfig,
    name='mnist',
    image_size=(28, 28),
    num_classes=10,
    num_samples=60_000,
    split_content='train',
)
MnistTestConfig = functools.partial(
    _DatasetConfig,
    name='mnist',
    image_size=(28, 28),
    num_classes=10,
    num_samples=10_000,
    split_content='test',
)
