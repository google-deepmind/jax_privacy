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

"""Data augmentation with augmult (Hoffer et al., 2019; Fort et al., 2021).

References:
  E. Hoffer, T. Ben-Nun, I. Hubara, N. Giladi, T. Hoefler, and D. Soudry.
  Augment your batch: bettertraining with larger batches.arXiv, 2019.
  S. Fort, A. Brock, R. Pascanu, S. De, and S. L. Smith.
  Drawing multiple augmentation samples perimage during training efficiently
  decreases test error.arXiv, 2021.
"""

from typing import Optional, Sequence, Tuple

import tensorflow.compat.v2 as tf


def apply_augmult(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: Sequence[int],
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    crop_size: Optional[Sequence[int]] = None,
    pad: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Augmult data augmentation (Hoffer et al., 2019; Fort et al., 2021).

  Args:
    image: (single) image to augment.
    label: label corresponding to the image (not modified by this function).
    image_size: new size for the image.
    augmult: number of augmentation multiplicities to use. This number should
      be non-negative (this function will fail if it is not).
    random_flip: whether to use random horizontal flips for data augmentation.
    random_crop: whether to use random crops for data augmentation.
    crop_size: size of the crop for random crops.
    pad: optional padding before the image is cropped.
  Returns:
    images: augmented images with a new prepended dimension of size `augmult`.
    labels: repeated labels with a new prepended dimension of size `augmult`.
  """
  image = tf.reshape(image, image_size)

  # No augmentations; return original images and labels with a new dimension.
  if augmult == 0:
    images = tf.expand_dims(image, axis=0)
    labels = tf.expand_dims(label, axis=0)
  # Perform one or more augmentations.
  elif augmult > 0:
    raw_image = tf.identity(image)
    augmented_images = []

    for _ in range(augmult):
      image_now = raw_image

      if random_crop:
        if pad:
          image_now = padding_input(image_now, pad=pad)
        image_now = tf.image.random_crop(
            image_now,
            size=(crop_size if crop_size is not None else image_size),
        )
      if random_flip:
        image_now = tf.image.random_flip_left_right(image_now)

      augmented_images.append(image_now)

    images = tf.stack(augmented_images, axis=0)
    labels = tf.stack([label]*augmult, axis=0)
  else:
    raise ValueError('Augmult should be non-negative.')

  return images, labels


def padding_input(x: tf.Tensor, pad: int):
  """Pad input image through 'mirroring' on the four edges.

  Args:
    x: image to pad.
    pad: number of padding pixels.
  Returns:
    Padded image.
  """
  x = tf.concat([x[:pad, :, :][::-1], x, x[-pad:, :, :][::-1]], axis=0)
  x = tf.concat([x[:, :pad, :][:, ::-1], x, x[:, -pad:, :][:, ::-1]], axis=1)
  return x
