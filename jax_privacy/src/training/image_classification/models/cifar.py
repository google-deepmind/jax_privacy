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

"""Definition of the CIFAR Wide Residual Network."""
import functools

import haiku as hk
import haiku.initializers as hk_init
import jax.numpy as jnp

from jax_privacy.src.training.image_classification.models import common


class WideResNet(hk.Module):
  """A Module defining a Wide ResNet."""

  CONV_MODULES = {
      'WSConv2D': common.WSConv2D,
      'Conv2D': hk.Conv2D,
  }

  def __init__(
      self,
      num_classes: int = 10,
      depth: int = 16,
      width: int = 4,
      dropout_rate: float = 0.0,
      use_skip_init: bool = False,
      use_skip_paths: bool = True,
      which_conv: str = 'WSConv2D',  # Conv2D or WSConv2D.
      which_norm: str = 'GroupNorm',  # LayerNorm / GroupNorm.
      activation: str = 'relu',
      groups: int = 16,  # Only used for GroupNorm.
  ):
    super().__init__()
    self.num_output_classes = num_classes
    self.width = width

    if which_norm is None:
      self.norm_fn = lambda *args, **kwargs: (lambda array: array)
    else:
      self.norm_fn = getattr(hk, which_norm)
      if which_norm == 'GroupNorm':
        self.norm_fn = functools.partial(self.norm_fn, groups=groups)

    self.conv_fn = self.CONV_MODULES[which_conv]
    if which_conv == 'WSConv2D':
      self.conv_fn = functools.partial(
          self.conv_fn, w_init=hk_init.VarianceScaling(1.0))

    self.use_skip_init = use_skip_init
    self.use_skip_paths = use_skip_paths
    self.dropout_rate = dropout_rate
    self.resnet_blocks = (depth - 4) // 6
    self.activation = common.activations_dict[activation]

  @hk.transparent
  def apply_skip_init(self, net, name):
    scale = hk.get_parameter(name, [1], init=jnp.zeros)
    return net * scale

  @hk.transparent
  def residual_block(self, net, width, strides, name):
    """Creates a residual block."""
    for i in range(self.resnet_blocks):
      if self.use_skip_paths:
        # This is the 'skip' branch.
        skip = net
        if i == 0:
          skip = self.activation(skip)
          skip = self.norm_fn(name=name + '_skip_norm')(skip)
          skip = self.conv_fn(
              width,
              name=name + '_skip_conv',
              stride=strides,
              kernel_shape=(1, 1),
          )(skip)
      # This is the 'residual' branch.
      for j in range(2):
        name_suffix = str(i) + '_' + str(j)
        strides = strides if name_suffix == '0_0' else (1, 1)
        net = self.activation(net)
        net = self.norm_fn(name=name + '_norm_' + name_suffix)(net)
        net = self.conv_fn(
            width,
            name=name + 'Conv_' + name_suffix,
            kernel_shape=(3, 3),
            stride=strides,
        )(net)
      # Merge both branches.
      if self.use_skip_init:
        net = self.apply_skip_init(net, name=name + 'Scale_' + name_suffix)
      if self.use_skip_paths:
        net += skip
    return net

  def __call__(self, inputs, is_training):
    net = self.conv_fn(16, name='First_conv', kernel_shape=(3, 3))(inputs)
    net = self.residual_block(
        net, width=16 * self.width, strides=(1, 1), name='Block_1')
    net = self.residual_block(
        net, width=32 * self.width, strides=(2, 2), name='Block_2')
    net = self.residual_block(
        net, width=64 * self.width, strides=(2, 2), name='Block_3')
    net = self.activation(net)

    net = self.norm_fn(name='Final_norm')(net)

    net = jnp.mean(net, axis=[1, 2], dtype=jnp.float32)

    if self.dropout_rate > 0.0:
      dropout_rate = self.dropout_rate if is_training else 0.0
      net = hk.dropout(hk.next_rng_key(), dropout_rate, net)

    return hk.Linear(
        self.num_output_classes,
        w_init=hk_init.VarianceScaling(1.0),
        name='Softmax',
    )(net)
