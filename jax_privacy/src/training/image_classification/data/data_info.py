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

"""Data information."""

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass(frozen=True)
class Split:

  num_samples: int
  split_content: Optional[Any] = None


_all_dataset_info: Dict[str, Dict[str, Any]] = {
    'cifar10': {
        'num_classes': 10,
        'train': Split(
            num_samples=45_000,
            split_content='train[:45000]',
        ),
        'valid': Split(
            num_samples=5_000,
            split_content='train[45000:]',
        ),
        'train_valid': Split(
            num_samples=50_000,
            split_content='train',
        ),
        'test': Split(
            num_samples=10_000,
            split_content='test',
        ),
    },
    'cifar100': {
        'num_classes': 100,
        'train': Split(
            num_samples=45_000,
            split_content='train[:45000]',
        ),
        'valid': Split(
            num_samples=5_000,
            split_content='train[45000:]',
        ),
        'train_valid': Split(
            num_samples=50_000,
            split_content='train',
        ),
        'test': Split(
            num_samples=10_000,
            split_content='test',
        ),
    },
    'svhn_cropped': {
        'num_classes': 10,
        'train': Split(
            num_samples=68_257,
            split_content='train[:68257]',
        ),
        'valid': Split(
            num_samples=5_000,
            split_content='train[68257:]',
        ),
        'train_valid': Split(
            num_samples=73_257,
            split_content='train',
        ),
        'test': Split(
            num_samples=26_032,
            split_content='test',
        ),
    },
    'imagenet': {
        'num_classes': 1000,
        'train': Split(
            num_samples=1_271_167,
            split_content='train[10000:]',
        ),
        'valid': Split(
            num_samples=10_000,
            split_content='train[:10000]',
        ),
        'train_valid': Split(
            num_samples=1_281_167,
            split_content='train',
        ),
        'test': Split(
            num_samples=50_000,
            split_content='validation',
        ),
    },
    'places365': {
        'num_classes': 365,
        'train': Split(
            num_samples=1_803_460,
            split_content='train',
        ),
        'valid': Split(
            num_samples=36_500,
            split_content='validation',
        ),
        'train_valid': Split(
            num_samples=1_839_960,
            split_content='train_valid',
        ),
        'test': Split(
            num_samples=328_500,
            split_content='test',
        ),
    },
}


@dataclasses.dataclass(frozen=True)
class Dataset:

  name: str
  num_classes: int
  train: Split
  eval: Split


def get_dataset(name: str, train_split: str, eval_split: str) -> Dataset:
  if train_split not in ('train', 'train_valid'):
    raise ValueError(f'Invalid train split: {train_split}.')
  return Dataset(
      name=name,
      num_classes=_all_dataset_info[name]['num_classes'],
      train=_all_dataset_info[name][train_split],
      eval=_all_dataset_info[name][eval_split],
  )


def num_samples(name: str, split: str) -> int:
  if split not in ('train', 'train_valid', 'valid', 'test'):
    raise ValueError(f'Invalid split: {split}.')
  return _all_dataset_info[name][split].num_samples
