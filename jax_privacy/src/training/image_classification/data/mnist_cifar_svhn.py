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

"""Data loading functions for MNIST / CIFAR / SVHN."""

import functools
from typing import Iterator, Optional, Tuple

import chex
import jax.numpy as jnp
from jax_privacy.src.training.image_classification.data import data_info
from jax_privacy.src.training.image_classification.data import image_dataset_loader
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def build_train_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_train: Tuple[int, int],
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    batch_size_per_device_per_step: int,
) -> Iterator[Tuple[chex.Array, chex.Array]]:
  """Builds the training input pipeline for MNIST / SVHN / CIFAR.

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
    Iterator of (images, labels) pairs of training samples.
  """
  ds = tfds.load(
      name=dataset.name,
      split=dataset.train.split_content,
      as_supervised=True,
  )
  preprocess_fn = functools.partial(
      preprocess_batch,
      is_training=True,
      image_resize=image_size_train,
      augmult=augmult,
      random_crop=random_crop,
      random_flip=random_flip,
      dataset=dataset,
  )
  return image_dataset_loader.load_32x32_image_dataset(
      ds,
      is_training=True,
      batch_size_per_device_per_step=batch_size_per_device_per_step,
      preprocess_fn=preprocess_fn,
  )


def build_eval_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_eval: Tuple[int, int],
    batch_size_eval: int,
) -> Iterator[Tuple[chex.Array, chex.Array]]:
  """Builds the evaluation input pipeline for MNIST / SVHN / CIFAR.

  Args:
    dataset: dataset to load.
    image_size_eval: size of the images at evaluation time.
    batch_size_eval: batch-size for the evaluation.
  Returns:
    Iterator of (images, labels) pairs of evaluation samples.
  """
  ds = tfds.load(
      name=dataset.name,
      split=dataset.eval.split_content,
      as_supervised=True,
  )
  preprocess_fn = functools.partial(
      preprocess_batch,
      is_training=False,
      image_resize=image_size_eval,
      augmult=0,
      random_crop=False,
      random_flip=False,
      dataset=dataset,
  )
  return image_dataset_loader.load_32x32_image_dataset(
      ds,
      is_training=False,
      batch_size_per_device_per_step=batch_size_eval,
      preprocess_fn=preprocess_fn,
  )


def preprocess_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    *,
    is_training: bool,
    image_resize: Optional[int],
    dataset: data_info.Dataset,
    augmult: int,
    random_flip: bool,
    random_crop: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Pre-processing module."""
  if dataset.name == 'mnist':
    return _preprocess_batch_mnist(images, labels)
  elif dataset.name in ('cifar10', 'cifar100', 'svhn_cropped'):
    return image_dataset_loader.preprocess_32x32(
        images,
        labels,
        is_training=is_training,
        normalization_fn=normalize_cifar_svhn,
        augmult=augmult,
        random_flip=random_flip,
        random_crop=random_crop,
        num_classes=dataset.num_classes,
        image_resize=image_resize,
    )
  else:
    raise ValueError(f'Invalid dataset {dataset.name}.')


def _preprocess_batch_mnist(
    images: tf.Tensor,
    labels: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Pre-processing module."""
  images = tf.image.convert_image_dtype(images, tf.float32)
  images = tf.clip_by_value(images * 2. - 1., -1., 1.)
  labels = tf.squeeze(tf.one_hot(labels, 10))
  return images, labels


def normalize_cifar_svhn(
    images: chex.Array,
) -> chex.Array:
  """Center and standardize images."""
  means = jnp.array([0.49139968, 0.48215841, 0.44653091])
  stds = jnp.array([0.24703223, 0.24348513, 0.26158784])
  return (images - means) / stds
