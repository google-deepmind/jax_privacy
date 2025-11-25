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

"""Data loading."""
# pylint:disable=g-multiple-import
# pylint:disable=g-importing-member

from image_data import chexpert
from image_data import mimic_cxr

from image_data.augmult import AugmultConfig
from image_data.base import (
    DataInputs,
    DatasetConfig,
    ImageDatasetConfig,
)
from image_data.chexpert import (
    AbstractChexpertLoader,
    ChexpertTestInternalConfig,
    ChexpertTestOfficialConfig,
    ChexpertTrainInternalConfig,
    ChexpertTrainOfficialConfig,
    ChexpertValidInternalConfig,
    ChexpertValidOfficialConfig,
)
from image_data.imagenet import (
    ImageNetLoader,
    ImageNetConfig,
    ImagenetTestConfig,
    ImagenetTrainConfig,
    ImagenetTrainValidConfig,
    ImagenetValidConfig,
    ImageNetNumSamples,
)
from image_data.loader import (
    DataLoader,
    LoadRawDataTfFn,
    LoadDatasetFn,
    default_load_dataset,
    grain_load_dataset,
)
from image_data.mimic_cxr import (
    AbstractMimicCxrLoader,
    MimicCxrTestInternalConfig,
    MimicCxrTestOfficialConfig,
    MimicCxrTrainInternalConfig,
    MimicCxrTrainOfficialConfig,
    MimicCxrValidInternalConfig,
    MimicCxrValidOfficialConfig,
)
from image_data.mnist_cifar_svhn import (
    MnistLoader,
    Cifar10Loader,
    Cifar100Loader,
    SvhnLoader,
    Cifar10TrainConfig,
    Cifar10TrainValidConfig,
    Cifar10ValidConfig,
    Cifar10TestConfig,
    Cifar100TrainConfig,
    Cifar100TrainValidConfig,
    Cifar100ValidConfig,
    Cifar100TestConfig,
    SvhnTrainConfig,
    SvhnValidConfig,
    SvhnTrainValidConfig,
    SvhnTestConfig,
    MnistTrainConfig,
    MnistValidConfig,
    MnistTrainValidConfig,
    MnistTestConfig,
)
from image_data.places365 import (
    Places365Loader,
    Places365TrainConfig,
    Places365ValidConfig,
    Places365TrainValidConfig,
    Places365Testconfig,
    Places365NumSamples,
)


MULTILABEL_DATASETS = (
    chexpert.ChexpertConfig,
    mimic_cxr.MimicCxrConfig,
)
