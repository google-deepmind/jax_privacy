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

"""ImageNet dataset with typical pre-processing and data augmentations."""

import dataclasses
import enum
import functools

from jax_privacy.experiments.image_data import base
from jax_privacy.experiments.image_data import loader
import tensorflow as tf
import tensorflow_datasets as tfds


MEAN_RGB = (0.485, 0.456, 0.406)
STDDEV_RGB = (0.229, 0.224, 0.225)


class ImageNetNumSamples(enum.IntEnum):

  TRAIN = 1_271_167
  VALID = 10_000
  TEST = 50_000


@dataclasses.dataclass(kw_only=True, slots=True)
class ImageNetConfig(base.ImageDatasetConfig):
  """ImageNet dataset.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train[:50000]".
    name: Unique identifying name for the dataset.
    preprocess_name: Name of preprocessing function. The default is to
      `standardise` the images to preserve the current behaviour in image
      classification. Going forward, consider setting this to `center` to
      avoid data-dependent pre-processing.
  """
  num_samples: int
  split_content: str
  image_size: tuple[int, int]
  num_classes: int = 1000
  name: str = 'imagenet'
  preprocess_name: str = 'standardise'

  def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
    """Normalizes the input image."""
    if self.preprocess_name == 'center':
      return base.center_image(image)
    elif self.preprocess_name == 'standardise':
      return base.standardize_image_per_channel(
          image,
          mean_per_channel=MEAN_RGB,
          stddev_per_channel=STDDEV_RGB,
      )
    else:
      raise NotImplementedError()

  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input label."""
    return tf.one_hot(label, depth=self.num_classes)


@dataclasses.dataclass(kw_only=True, slots=True)
class ImageNetLoader(loader.DataLoader):
  """Data loader for ImageNet."""

  config: ImageNetConfig

  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    if self.config.using_large_images:
      ds = tfds.load(
          'imagenet2012:5.*.*',
          split=self.config.split_content,
          decoders={'image': tfds.decode.SkipDecoding()},
          shuffle_files=shuffle_files,
      )
    else:
      im_size = self.config.image_size
      ds = tfds.load(
          name=f'imagenet_resized/{im_size[0]}x{im_size[1]}',
          split=self.config.split_content,
          shuffle_files=shuffle_files,
      )
    return ds.map(base.DataInputs.from_dict)


ImagenetTrainConfig = functools.partial(
    ImageNetConfig,
    num_samples=ImageNetNumSamples['TRAIN'],
    split_content='train[10000:]',
)
ImagenetValidConfig = functools.partial(
    ImageNetConfig,
    num_samples=ImageNetNumSamples['VALID'],
    split_content='train[:10000]',
)
ImagenetTrainValidConfig = functools.partial(
    ImageNetConfig,
    num_samples=(ImageNetNumSamples['TRAIN'] + ImageNetNumSamples['VALID']),
    split_content='train',
)
ImagenetTestConfig = functools.partial(
    ImageNetConfig,
    num_samples=ImageNetNumSamples['TEST'],
    split_content='validation',
)
