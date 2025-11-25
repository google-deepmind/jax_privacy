# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
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
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import random
from typing import Any, Type

import chex
import jax
from image_data import augmult
from image_data import decoder
import numpy as np
import tensorflow as tf


TensorOrArray = tf.Tensor | chex.Array


@chex.dataclass(frozen=True)
class DataInputs:
  """Data inputs (either as a single example or as a batch).

  Attributes:
    image: Image content (potentially batched).
    label: Label content (potentially batched).
    metadata: Auxiliary content (potentially batched).
  """

  image: TensorOrArray
  label: TensorOrArray
  metadata: Mapping[str, Any] = dataclasses.field(  # pylint: disable=invalid-field-call
      default_factory=dict)

  @classmethod
  def from_dict(
      cls: Type['DataInputs'],
      data_dict: Mapping[str, TensorOrArray],
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
    seed: (Optional) Random seed for the dataset (used for shuffling/sampling).
  """

  num_samples: int
  num_classes: int
  name: str
  split_content: str
  seed: int | None = None

  def __post_init__(self):
    if self.seed is None:
      self.seed = random.randint(0, 2**32-1)

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
  def _normalize_image(self, image: TensorOrArray) -> TensorOrArray:
    """Normalizes the input image."""

  @abc.abstractmethod
  def _preprocess_label(self, label: TensorOrArray) -> TensorOrArray:
    """Pre-processes the input label."""

  def _rescale_image(
      self,
      image: TensorOrArray,
  ) -> TensorOrArray:
    """Converts the image to float and tescales to the [0, 1] range."""
    if isinstance(image, tf.Tensor):
      return tf.image.convert_image_dtype(image, np.float32)
    else:
      return np.float32(image) / 255

  def _resize_image(
      self,
      image: TensorOrArray,
      *,
      is_training: bool,
  ) -> TensorOrArray:
    """Resizes the image to the desired resolution."""
    if isinstance(image, tf.Tensor):
      return tf.image.resize(image, self.image_size)
    else:
      if is_training:
        # Add augmult and channel dimensions.
        image_size = (
            image.shape[0],
            self.image_size[0],
            self.image_size[1],
            image.shape[-1],
        )
      else:
        # Add channel dimension.
        image_size = (
            self.image_size[0],
            self.image_size[1],
            image.shape[-1],
        )
      return jax.image.resize(image, image_size, method='bilinear')

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
    use_tf = isinstance(image, tf.Tensor)

    if self.using_large_images:
      if use_tf:
        # Large images are decoded in a custom resolution (either `image_size`,
        # or slightly larger than `image_size` if using random crops).
        image = decoder.decode_large_image(
            image,
            image_size=self.image_size,
            augmult_config=augmult_config,
        )
      else:
        # TODO: b/409948867 - Implement large image decoding in pure python.
        raise NotImplementedError('Large images require TensorFlow (for now).')
    else:
      # Otherwise, the image is simply converted to float in its original
      # resolution and scaled to [0, 1].
      image = self._rescale_image(image)

    image = self._normalize_image(image)
    label = self._preprocess_label(label)

    if is_training and augmult_config is not None:
      # Apply augmult in training mode.
      if use_tf:
        image, label = augmult_config.apply(
            image=image,
            label=label,
            crop_size=[*self.image_size, image.shape[2]],
        )
      else:
        # TODO: b/409948867 - Implement augmult in pure python.
        raise NotImplementedError(
            'Augmult requires TensorFlow (for now).'
        )
    elif is_training:
      # Match the augmult dimensions in training mode.
      if use_tf:
        expand_dims_fn = tf.expand_dims
      else:
        expand_dims_fn = np.expand_dims
      image = expand_dims_fn(image, axis=0)
      label = expand_dims_fn(label, axis=0)
    else:
      # Ignore augmult in evaluation mode.
      pass

    if not self.using_large_images and self.image_size:
      # Small images may get resized after the pre-processing and data
      # augmentation.
      image = self._resize_image(image, is_training=is_training)

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
    image: TensorOrArray,
    min_value: float = -1.0,
    max_value: float = 1.0,
) -> TensorOrArray:
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
    image: TensorOrArray,
    mean_per_channel: float | tuple[float, float, float],
    stddev_per_channel: float | tuple[float, float, float],
) -> TensorOrArray:
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
  if isinstance(image, tf.Tensor):
    constant_fn = functools.partial(
        tf.constant, shape=[1, 1, image.shape[2]], dtype=image.dtype
    )
  else:
    constant_fn = lambda x: np.expand_dims(
        np.array(x, dtype=image.dtype), axis=(0, 1)
    )

  mean_per_channel = constant_fn(mean_per_channel)
  stddev_per_channel = constant_fn(stddev_per_channel)

  return (image - mean_per_channel) / stddev_per_channel


def one_hot_label(
    label: TensorOrArray,
    num_classes: int,
) -> TensorOrArray:
  """Converts a label to a one-hot encoding."""
  if isinstance(label, tf.Tensor):
    return tf.one_hot(label, depth=num_classes)
  else:
    return jax.nn.one_hot(label, num_classes=num_classes)
