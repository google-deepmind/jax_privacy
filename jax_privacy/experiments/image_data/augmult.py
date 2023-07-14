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

"""Data augmentation with augmult (Hoffer et al., 2019; Fort et al., 2021).

References:
  E. Hoffer, T. Ben-Nun, I. Hubara, N. Giladi, T. Hoefler, and D. Soudry.
  Augment your batch: bettertraining with larger batches.arXiv, 2019.
  S. Fort, A. Brock, R. Pascanu, S. De, and S. L. Smith.
  Drawing multiple augmentation samples perimage during training efficiently
  decreases test error.arXiv, 2021.
"""

import dataclasses
from typing import Sequence

import tensorflow.compat.v2 as tf


@dataclasses.dataclass(kw_only=True, slots=True)
class AugmultConfig:
  """Preprocessing options for images at training time.

  Attributes:
    augmult: Number of augmentation multiplicities to use. `augmult=0`
      corresponds to no augmentation at all, `augmult=1` to standard data
      augmentation (one augmented view per mini-batch) and `augmult>1` to
      having several augmented view of each sample within the mini-batch.
    random_crop: Whether to use random crops for data augmentation.
    random_flip: Whether to use random horizontal flips for data augmentation.
    random_color: Whether to use random color jittering for data augmentation.
    pad: Optional padding before the image is cropped for data augmentation.
  """
  augmult: int
  random_crop: bool
  random_flip: bool
  random_color: bool
  pad: int | None = 4

  def apply(
      self,
      image: tf.Tensor,
      label: tf.Tensor,
      *,
      crop_size: Sequence[int],
  ) -> tuple[tf.Tensor, tf.Tensor]:
    return apply_augmult(
        image,
        label,
        augmult=self.augmult,
        random_flip=self.random_flip,
        random_crop=self.random_crop,
        random_color=self.random_color,
        pad=self.pad,
        crop_size=crop_size,
    )


def apply_augmult(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    random_color: bool,
    crop_size: Sequence[int],
    pad: int | None,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Augmult data augmentation (Hoffer et al., 2019; Fort et al., 2021).

  Args:
    image: (single) image to augment.
    label: label corresponding to the image (not modified by this function).
    augmult: number of augmentation multiplicities to use. This number
      should be non-negative (this function will fail if it is not).
    random_flip: whether to use random horizontal flips for data augmentation.
    random_crop: whether to use random crops for data augmentation.
    random_color: whether to use random color jittering for data augmentation.
    crop_size: size of the crop for random crops.
    pad: optional padding before the image is cropped.
  Returns:
    images: augmented images with a new prepended dimension of size `augmult`.
    labels: repeated labels with a new prepended dimension of size `augmult`.
  """
  if augmult == 0:
    # No augmentations; return original images and labels with a new dimension.
    images = tf.expand_dims(image, axis=0)
    labels = tf.expand_dims(label, axis=0)
  elif augmult > 0:
    # Perform one or more augmentations.
    raw_image = tf.identity(image)
    augmented_images = []

    for _ in range(augmult):
      image_now = raw_image

      if random_crop:
        if pad:
          image_now = padding_input(image_now, pad=pad)
        image_now = tf.image.random_crop(image_now, size=crop_size)
      if random_flip:
        image_now = tf.image.random_flip_left_right(image_now)
      if random_color:
        # values copied/adjusted from a color jittering tutorial
        # https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
        image_now = tf.image.random_hue(image_now, 0.1)
        image_now = tf.image.random_saturation(image_now, 0.6, 1.6)
        image_now = tf.image.random_brightness(image_now, 0.15)
        image_now = tf.image.random_contrast(image_now, 0.7, 1.3)

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
