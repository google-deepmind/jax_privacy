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

"""Data loading functions."""

from typing import Dict, Iterator, Tuple

import chex
from jax_privacy.src.training.image_classification.data import data_info
from jax_privacy.src.training.image_classification.data import imagenet
from jax_privacy.src.training.image_classification.data import mnist_cifar_svhn
from jax_privacy.src.training.image_classification.data import places365


def build_train_input(
    *,
    dataset: data_info.Dataset,
    image_size_train: Tuple[int, int],
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    batch_size_per_device_per_step: int,
) -> Iterator[Dict[str, chex.Array]]:
  """Builds the training input pipeline for the specified dataset.

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
  if dataset.name.lower() in ('cifar10', 'cifar100', 'mnist', 'svhn_cropped'):
    return mnist_cifar_svhn.build_train_input_dataset(
        dataset=dataset,
        image_size_train=image_size_train,
        augmult=augmult,
        random_crop=random_crop,
        random_flip=random_flip,
        batch_size_per_device_per_step=batch_size_per_device_per_step,
    )
  elif dataset.name.lower() == 'imagenet':
    return imagenet.build_train_input_dataset(
        dataset=dataset,
        image_size_train=image_size_train,
        augmult=augmult,
        random_crop=random_crop,
        random_flip=random_flip,
        batch_size_per_device_per_step=batch_size_per_device_per_step,
    )
  elif dataset.name.lower() == 'places365':
    return places365.build_train_input_dataset(
        dataset=dataset,
        image_size_train=image_size_train,
        augmult=augmult,
        random_crop=random_crop,
        random_flip=random_flip,
        batch_size_per_device_per_step=batch_size_per_device_per_step,
    )
  else:
    raise ValueError(f'Invalid dataset: {dataset.name}.')


def build_eval_input(
    *,
    dataset: data_info.Dataset,
    image_size_eval: Tuple[int, int],
    batch_size_eval: int,
) -> Iterator[Dict[str, chex.Array]]:
  """Builds the evaluation input pipeline for the specified dataset.

  Args:
    dataset: dataset to load.
    image_size_eval: size of the images at evaluation time.
    batch_size_eval: batch-size for the evaluation.
  Returns:
    Iterator of pairs of evaluation samples with format
    `{'images': images, 'labels': labels}`.
  """
  if dataset.name.lower() in ('cifar10', 'cifar100', 'mnist', 'svhn_cropped'):
    return mnist_cifar_svhn.build_eval_input_dataset(
        dataset=dataset,
        image_size_eval=image_size_eval,
        batch_size_eval=batch_size_eval,
    )
  elif dataset.name.lower() == 'imagenet':
    return imagenet.build_eval_input_dataset(
        dataset=dataset,
        image_size_eval=image_size_eval,
        batch_size_eval=batch_size_eval,
    )
  elif dataset.name.lower() == 'places365':
    return places365.build_eval_input_dataset(
        dataset=dataset,
        image_size_eval=image_size_eval,
        batch_size_eval=batch_size_eval,
    )
  else:
    raise ValueError(f'Invalid dataset: {dataset.name}.')
