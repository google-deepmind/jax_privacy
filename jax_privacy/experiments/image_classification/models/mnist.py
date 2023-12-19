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

"""Two-layer CNN for MNIST (mainly for membership inference attacks)."""

import dataclasses
import functools

import haiku as hk
import jax
from jax_privacy.experiments.image_classification.models import base
from jax_privacy.experiments.image_classification.models import common


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class MnistCnnConfig(base.ModelConfig):
  """Config for Mnist CNN."""

  activation: common.Activation = common.Activation.RELU

  def make(self, num_classes: int) -> base.Model:
    return base.Model.from_hk_module(
        MnistCNN,
        num_classes=num_classes,
        activation=self.activation,
    )


class MnistCNN(hk.Module):
  """Hard-coded two-layer CNN."""

  def __init__(
      self,
      num_classes: int = 10,
      activation: common.Activation = common.Activation.RELU,
  ):
    super().__init__()

    # All conv layers have a kernel shape of 3 and a stride of 1.
    self._conv_1 = hk.Conv2D(
        output_channels=16,
        kernel_shape=8,
        stride=2,
        padding='SAME',
        name='conv2d_1',
    )
    self._conv_2 = hk.Conv2D(
        output_channels=32,
        kernel_shape=4,
        stride=2,
        padding='VALID',
        name='conv2d_2',
    )

    # First linear layer.
    self._linear = hk.Linear(32, name='linear')

    # Classification layer.
    self._logits_module = hk.Linear(num_classes, name='linear_1')
    self._pool = functools.partial(
        hk.max_pool,
        window_shape=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
    )

    self._activation = activation

  def __call__(self, inputs: jax.Array, is_training: bool) -> jax.Array:
    return hk.Sequential([
        self._conv_1,
        self._activation.fn,
        self._pool,
        self._conv_2,
        self._activation.fn,
        self._pool,
        hk.Flatten(),
        self._linear,
        self._activation.fn,
        self._logits_module,
    ])(inputs)
