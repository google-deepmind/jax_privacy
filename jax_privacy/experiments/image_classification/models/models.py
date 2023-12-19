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

"""Imports for all models."""

import enum
import functools

from jax_privacy.experiments.image_classification.models import base
from jax_privacy.experiments.image_classification.models import cifar
from jax_privacy.experiments.image_classification.models import imagenet_nfnets
from jax_privacy.experiments.image_classification.models import imagenet_nfresnets
from jax_privacy.experiments.image_classification.models import mnist


class Registry(enum.Enum):
  """Model registry."""

  # CIFAR models.
  WRN_40_4_CIFAR100 = 'wrn_40_4_cifar100'

  # ImageNet-1k-32 models.
  WRN_40_4_IMAGENET32 = 'wrn_40_4_imagenet32'
  WRN_28_10_IMAGENET32 = 'wrn_28_10_imagenet32'

  # ImageNet-21k models.
  NFNET_F0_IM21K = 'nfnet_f0_im21k'
  NFNET_F1_IM21K = 'nfnet_f1_im21k'
  NFNET_F3_IM21K = 'nfnet_f3_im21k'

  @property
  def path(self) -> str:
    return self.value


# Aliasing
Model = base.Model
ModelConfig = base.ModelConfig
WithRestoreModelConfig = base.WithRestoreModelConfig
WideResNetConfig = cifar.WideResNetConfig
NFNetConfig = imagenet_nfnets.NFNetConfig
nfnet_config = imagenet_nfnets.nfnet_config
NFResNetConfig = imagenet_nfresnets.NFResNetConfig
MnistCnnConfig = mnist.MnistCnnConfig

nfnet_f0_config = functools.partial(nfnet_config, variant='F0')
nfnet_f1_config = functools.partial(nfnet_config, variant='F1')
nfnet_f3_config = functools.partial(nfnet_config, variant='F3')
