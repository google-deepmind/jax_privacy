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

"""Places 365 dataset with typical pre-processing and advanced augs."""

from typing import Dict, Iterator, Tuple

import chex
import jax
from jax_privacy.src.training.image_classification.data import data_info
from jax_privacy.src.training.image_classification.data import image_dataset_loader
import tensorflow as tf
import tensorflow_datasets as tfds


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def build_train_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_train: Tuple[int, int],
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    batch_size_per_device_per_step: int,
) -> Iterator[Dict[str, chex.Array]]:
  """Builds the training input pipeline for the Places dataset.

  Args:
    dataset: dataset to load.
    image_size_train: size of the images at training time.
    augmult: number of augmentation multiplicities to use. `augmult=0`
      corresponds to no augmentation at all, `augmult=1` to standard data
      augmentation (one augmented view per mini-batch) and `augmult>1` to having
      several augmented view of each sample within the mini-batch.
    random_crop: whether to use random crops for data augmentation.
    random_flip: whether to use random horizontal flips for data augmentation.
    batch_size_per_device_per_step: batch-size to fit on each device at every
      iteration. Note that if e.g. `batch_size_per_device_per_step=16` and
      `augmult=8`, each device will effectively use 8*16 samples at each
      iteration.
  Returns:
    Iterator of pairs of training samples with format
    `{'images': images, 'labels': labels}`.
  """
  assert dataset.name == 'places365'

  ds = tfds.load(
      'places365_small:2.*.*',
      split=dataset.train.split_content,
      decoders={'image': tfds.decode.SkipDecoding()},
      as_supervised=True,
  )

  return image_dataset_loader.load_image_dataset(
      ds=ds,
      num_samples=dataset.train.num_samples,
      normalize_fn=_normalize_image,
      is_training=True,
      batch_dims=(jax.local_device_count(), batch_size_per_device_per_step),
      image_size=image_size_train,
      augmult=augmult,
      random_crop=random_crop,
      random_flip=random_flip,
      label_preprocess_fn=(
          lambda x: tf.one_hot(x, dataset.num_classes)),
    )


def build_eval_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_eval: Tuple[int, int],
    batch_size_eval: int,
) -> Iterator[Dict[str, chex.Array]]:
  """Builds the evaluation input pipeline for the Places dataset.

  Args:
    dataset: dataset to load.
    image_size_eval: size of the images at evaluation time.
    batch_size_eval: batch-size for the evaluation.
  Returns:
    Iterator of pairs of evaluation samples with format
    `{'images': images, 'labels': labels}`.
  """
  assert dataset.name == 'places365'

  ds = tfds.load(
      'places365_small:2.*.*',
      split=dataset.eval.split_content,
      decoders={'image': tfds.decode.SkipDecoding()},
      as_supervised=True,
  )

  return image_dataset_loader.load_image_dataset(
      ds=ds,
      num_samples=dataset.eval.num_samples,
      normalize_fn=_normalize_image,
      is_training=False,
      batch_dims=(batch_size_eval,),
      image_size=image_size_eval,
      label_preprocess_fn=(
          lambda x: tf.one_hot(x, dataset.num_classes)),
    )
