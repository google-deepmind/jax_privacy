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

"""Places 365 dataset with typical pre-processing and advanced augs."""

import dataclasses
import enum
import functools
from typing import Sequence

from jax_privacy.experiments.image_data import base
from jax_privacy.experiments.image_data import loader
import tensorflow as tf
import tensorflow_datasets as tfds


MEAN_RGB = (0.485, 0.456, 0.406)
STDDEV_RGB = (0.229, 0.224, 0.225)


class Places365NumSamples(enum.IntEnum):

  TRAIN = 1_793_460
  VALID = 10_000
  TEST = 36_500


@dataclasses.dataclass(kw_only=True, slots=True)
class Places365Config(base.ImageDatasetConfig):
  """Builds the input pipeline for Places365.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    name: Unique identifying name for the dataset.
    split_content: Subset split, e.g. "train[:50000]".
    image_size: Image resolution.
    preprocess_name: Name of preprocessing function. The default is to
      `standardise` the images to preserve the current behaviour in image
      classification. Going forward, consider setting this to `center` to
      avoid data-dependent pre-processing.
    cached_class_names: names of the different classes stored in a cache to
      avoid multiple slow queries to TFDS.
  """
  name: str
  num_samples: int
  split_content: str
  num_classes: int = 365
  name: str = 'places365'
  image_size: tuple[int, int]
  preprocess_name: str = 'standardise'
  cached_class_names: Sequence[str] | None = dataclasses.field(
      default=None, init=False)

  @property
  def class_names(self) -> Sequence[str]:
    if self.cached_class_names is None:
      # This is relatively slow, so fetch the information only if required.
      self.cached_class_names = tfds.builder(
          'places365_small').info.features['label'].names
    return self.cached_class_names

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
      raise NotImplementedError(
          f'Preprocessing with {self.preprocess_name} not implemented for '
          'Places365.')

  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input label."""
    return tf.one_hot(label, depth=self.num_classes)


@dataclasses.dataclass(kw_only=True, slots=True)
class Places365Loader(loader.DataLoader):
  """Data loader for Places365."""

  config: Places365Config

  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    ds = tfds.load(
        'places365_small:2.*.*',
        split=self.config.split_content,
        decoders={'image': tfds.decode.SkipDecoding()},
        shuffle_files=shuffle_files,
    )
    return ds.map(base.DataInputs.from_dict)


Places365TrainConfig = functools.partial(
    Places365Config,
    num_samples=Places365NumSamples['TRAIN'],
    split_content='train[10000:]',
)
Places365ValidConfig = functools.partial(
    Places365Config,
    num_samples=Places365NumSamples['VALID'],
    split_content='train[:10000]',
)
Places365TrainValidConfig = functools.partial(
    Places365Config,
    num_samples=(Places365NumSamples['TRAIN'] + Places365NumSamples['VALID']),
    split_content='train',
)
Places365Testconfig = functools.partial(
    Places365Config,
    num_samples=Places365NumSamples['TEST'],
    split_content='validation',
)
