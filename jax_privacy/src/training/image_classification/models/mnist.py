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

"""Two-layer CNN for MNIST (mainly for membership inference attacks)."""
import functools

import chex
import haiku as hk
import jax

from jax_privacy.src.training.image_classification.models import common


class MnistCNN(hk.Module):
  """Hard-coded two-layer CNN."""

  def __init__(
      self,
      num_classes: int = 10,
      activation: common.Activation = jax.nn.relu
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

  def __call__(self, inputs: chex.Array, is_training: bool) -> chex.Array:
    return hk.Sequential([
        self._conv_1,
        self._activation,
        self._pool,
        self._conv_2,
        self._activation,
        self._pool,
        hk.Flatten(),
        self._linear,
        self._activation,
        self._logits_module,
    ])(inputs)
