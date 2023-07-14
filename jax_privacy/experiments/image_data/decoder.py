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

"""Image dataset loader with typical pre-processing and advanced augs."""

from jax_privacy.experiments.image_data import augmult
import numpy as np
import tensorflow as tf


def decode_large_image(
    image: tf.Tensor,
    *,
    image_size: tuple[int, int],
    augmult_config: augmult.AugmultConfig | None,
) -> tf.Tensor:
  """Decodes the image and returns it with float32 values within [0, 1]."""
  if image.dtype == tf.dtypes.string:
    image = _decode_and_center_crop(
        image_bytes=image,
    )
  # Convert to float32 and rescale to [0, 1] after the decoding.
  image = tf.image.convert_image_dtype(image, np.float32)

  if (augmult_config is not None
      and augmult_config.random_crop
      and augmult_config.augmult):
    # Increase the target size to later take random crops within the image,
    # e.g. 268x268 for 224x224 crops.
    image_size = [int(x * 1.2) for x in image_size]
  image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

  # NOTE: Bicubic resizes without clamping overshoots. This means values
  # returned will be outside the range [0.0, 1.0].
  return tf.clip_by_value(image, 0.0, 1.0)


def _decode_and_center_crop(
    *,
    image_bytes: tf.Tensor,
) -> tf.Tensor:
  """Decodes a JPEG and takes a square center crop to make the image square."""
  jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]
  min_size = tf.minimum(image_height, image_width)

  offset_height = (image_height - min_size) // 2
  offset_width = (image_width - min_size) // 2
  crop_window = tf.stack([offset_height, offset_width, min_size, min_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image  # dtype uint8
