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

"""Abstract class for a dataset split."""

import abc
import dataclasses
from typing import Any, Mapping, Sequence, Type

import chex
from jax_privacy.experiments.image_data import augmult
from jax_privacy.experiments.image_data import decoder
import numpy as np
import tensorflow as tf


@chex.dataclass(frozen=True)
class DataInputs:
  """Data inputs (either as a single example or as a batch).

  Attributes:
    image: Image content (potentially batched).
    label: Label content (potentially batched).
    metadata: Auxiliary content (potentially batched).
  """

  image: tf.Tensor | chex.Array
  label: tf.Tensor | chex.Array
  metadata: Mapping[str, Any] = dataclasses.field(  # pylint: disable=invalid-field-call
      default_factory=dict)

  @classmethod
  def from_dict(
      cls: Type['DataInputs'],
      data_dict: Mapping[str, tf.Tensor | chex.Array],
  ) -> 'DataInputs':
    metadata = {
        k: v for k, v in data_dict.items() if k not in ('image', 'label')}
    return cls(
        image=data_dict['image'],
        label=data_dict['label'],
        metadata=metadata,
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class DatasetConfig(metaclass=abc.ABCMeta):
  """Dataset configuration.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train[:50000]".
    name: Unique identifying name for the dataset.
  """

  num_samples: int
  num_classes: int
  name: str
  split_content: str

  @property
  def class_names(self) -> Sequence[str]:
    """Returns class names for the dataset, defaulting to an index value."""
    return [str(i) for i in range(self.num_classes)]

  @abc.abstractmethod
  def preprocess(
      self,
      data: DataInputs,
      *,
      is_training: bool,
      augmult_config: augmult.AugmultConfig | None = None,
  ) -> DataInputs:
    """Preprocesses the image and the label."""

  @abc.abstractmethod
  def make_fake_data(self) -> tf.data.Dataset:
    """Creates fake data for debugging purposes."""


@dataclasses.dataclass(kw_only=True, slots=True)
class ImageDatasetConfig(DatasetConfig):
  """Abstract dataset split using  loader for large images.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train[:50000]".
    name: Unique identifying name for the dataset.
    image_size: Image resolution to use.
    using_large_images: Whether to decode images as large images.
  """

  name: str
  num_samples: int
  num_classes: int
  split_content: str
  image_size: tuple[int, int]

  @property
  def using_large_images(self) -> bool:
    return min(self.image_size) > 64

  @abc.abstractmethod
  def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
    """Normalizes the input image."""

  @abc.abstractmethod
  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input label."""

  def preprocess(
      self,
      data: DataInputs,
      *,
      is_training: bool,
      augmult_config: augmult.AugmultConfig | None = None,
  ) -> DataInputs:
    """Preprocesses the image and the label."""
    if not is_training:
      # Ignore augmult in evaluation mode.
      augmult_config = None

    image, label = data.image, data.label

    if self.using_large_images:
      # Large images are decoded in a custom resolution (either `image_size`,
      # or slightly larger than `image_size` if using random crops).
      image = decoder.decode_large_image(
          image,
          image_size=self.image_size,
          augmult_config=augmult_config,
      )
    else:
      # Otherwise, the image is simply converted to float in its original
      # resolution and scaled to [0, 1].
      image = tf.image.convert_image_dtype(image, np.float32)

    image = self._normalize_image(image)
    label = self._preprocess_label(label)

    if is_training and augmult_config is not None:
      # Apply augmult in training mode.
      image, label = augmult_config.apply(
          image=image,
          label=label,
          crop_size=[*self.image_size, image.shape[2]],
      )
    elif is_training:
      # Match the augmult dimensions in training mode.
      image = tf.expand_dims(image, axis=0)
      label = tf.expand_dims(label, axis=0)
    else:
      # Ignore augmult in evaluation mode.
      pass

    if not self.using_large_images and self.image_size:
      # Small images may get resized after the pre-processing and data
      # augmentation.
      image = tf.image.resize(image, self.image_size)

    return DataInputs(image=image, label=label, metadata=data.metadata)

  def make_fake_data(self) -> tf.data.Dataset:
    """Creates fake data for debugging purposes."""
    fake_data = DataInputs(
        image=tf.random.normal(shape=(*self.image_size, 3)),
        label=tf.random.uniform(
            shape=(), minval=0, maxval=self.num_classes, dtype=tf.int32),
    )
    return tf.data.Dataset.from_tensors(fake_data).repeat(self.num_samples)


def center_image(
    image: tf.Tensor,
    min_value: float = -1.0,
    max_value: float = 1.0,
) -> tf.Tensor:
  """Centers the image to have values in [min_value, max_value].

  Args:
    image: A multi-dimensional array of floating point values in [0, 1].
    min_value: The minimum value for the pixels in the centered image.
    max_value: The minimum value for the pixels in the centered image.
  Returns:
    The centered image, with values in the range [min_value, max_value].
  """
  return image * (max_value - min_value) + min_value


def standardize_image_per_channel(
    image: tf.Tensor,
    mean_per_channel: float | tuple[float, float, float],
    stddev_per_channel: float | tuple[float, float, float],
) -> tf.Tensor:
  """Standardizes the image per channel.

  Args:
    image: A [H, W, C] array of floating point values in [0, 1].
    mean_per_channel: The mean value for pixels in each channel of the image.
    stddev_per_channel: The standard deviation for pixels in each channel of the
      image.

  Returns:
    The standardized image, with mean 0 and standard deviation 1 in each
    channel.
  """
  mean_per_channel = tf.constant(
      mean_per_channel, shape=[1, 1, image.shape[2]], dtype=image.dtype
  )
  stddev_per_channel = tf.constant(
      stddev_per_channel, shape=[1, 1, image.shape[2]], dtype=image.dtype
  )
  return (image - mean_per_channel) / stddev_per_channel
