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

"""ImageNet Norm-Free Residual Networks as defined in (Brock et al., 2021).

Reference:
  A. Brock, S. De, and S. L. Smith.
  Characterizing signal propagation to close the performance gap
  in unnormalized resnets.
  International Conference on Learning Representations, 2021.
"""

import dataclasses
from typing import Any, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.experiments.image_classification.models import base
from jax_privacy.experiments.image_classification.models import common


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class NFResNetConfig(base.ModelConfig):
  """Norm-Free preactivation ResNet config."""

  variant: str = 'ResNet50'
  width: int = 4
  alpha: float = 0.2
  stochdepth_rate: float = 0.1
  drop_rate: Optional[float] = None
  activation: common.Activation = common.Activation.SCALED_RELU
  fc_init: Any = None
  skipinit_gain: hk.initializers.Initializer = jnp.zeros
  use_se: bool = False
  se_ratio: float = 0.25
  name: str = 'NF_ResNet'

  def make(self, num_classes: int) -> base.Model:
    return base.Model.from_hk_module(
        NFResNet,
        num_classes=num_classes,
        variant=self.variant,
        width=self.width,
        alpha=self.alpha,
        stochdepth_rate=self.stochdepth_rate,
        drop_rate=self.drop_rate,
        activation=self.activation,
        fc_init=self.fc_init,
        skipinit_gain=self.skipinit_gain,
        use_se=self.use_se,
        se_ratio=self.se_ratio,
        name=self.name,
    )


class NFResNet(hk.Module):
  """Norm-Free preactivation ResNet."""

  variant_dict = {
      'ResNet50': {
          'depth': [3, 4, 6, 3]
      },
      'ResNet101': {
          'depth': [3, 4, 23, 3]
      },
      'ResNet152': {
          'depth': [3, 8, 36, 3]
      },
      'ResNet200': {
          'depth': [3, 24, 36, 3]
      },
      'ResNet288': {
          'depth': [24, 24, 24, 24]
      },
      'ResNet600': {
          'depth': [50, 50, 50, 50]
      },
  }

  def __init__(
      self,
      num_classes: int,
      *,
      variant: str = 'ResNet50',
      width: int = 4,
      alpha: float = 0.2,
      stochdepth_rate: float = 0.1,
      drop_rate: Optional[float] = None,
      activation: common.Activation = common.Activation.SCALED_RELU,
      fc_init: Any = None,
      skipinit_gain: hk.initializers.Initializer = jnp.zeros,
      use_se: bool = False,
      se_ratio: float = 0.25,
      name: str = 'NF_ResNet',
  ):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.variant = variant
    self.width = width
    block_params = self.variant_dict[self.variant]
    self.width_pattern = [item * self.width for item in [64, 128, 256, 512]]
    self.depth_pattern = block_params['depth']
    self.activation = activation
    if drop_rate is None:
      self.drop_rate = block_params.get('drop_rate', 0.0)
    else:
      self.drop_rate = drop_rate

    # Define the stem of the model.
    ch = int(16 * self.width)
    self.initial_conv = common.WSConv2D(
        ch,
        kernel_shape=7,
        stride=2,
        padding='SAME',
        with_bias=False,
        name='initial_conv')

    # Define the body of the model.
    self.blocks = []
    expected_std = 1.0
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    block_args = (self.width_pattern, self.depth_pattern, [1, 2, 2, 2])
    for block_width, stage_depth, stride in zip(*block_args, strict=True):
      for block_index in range(stage_depth):
        # Scalar pre-multiplier so each block sees an N(0,1) input at init.
        beta = 1. / expected_std
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        self.blocks += [
            NFResBlock(
                ch,
                block_width,
                stride=stride if block_index == 0 else 1,
                beta=beta,
                alpha=alpha,
                activation=self.activation,
                stochdepth_rate=block_stochdepth_rate,
                skipinit_gain=skipinit_gain,
                use_se=use_se,
                se_ratio=se_ratio,
            )
        ]
        ch = block_width
        index += 1
        # Reset expected std but still give it 1 block of growth.
        if block_index == 0:
          expected_std = 1.0
        expected_std = (expected_std**2 + alpha**2)**0.5

    # Define the head: by default, initialize with N(0, 0.01).
    if fc_init is None:
      fc_init = hk.initializers.RandomNormal(0.01, 0)
    self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)

  def __call__(self, x: jax.Array, is_training: bool) -> jax.Array:
    """Return the output of the final layer without any [log-]softmax."""
    # Forward through the stem.
    out = self.initial_conv(x)
    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
    # Forward through the blocks
    for block in self.blocks:
      out, unused_res_avg_var = block(out, is_training=is_training)
    # Final-conv->activation, pool, dropout, classify
    pool = jnp.mean(self.activation.fn(out), [1, 2])
    # Optionally apply dropout.
    if self.drop_rate > 0.0 and is_training:
      pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
    logits = self.fc(pool)
    return logits


class NFResBlock(hk.Module):
  """Normalizer-Free pre-activation ResNet Block."""

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      *,
      bottleneck_ratio: float = 0.25,
      kernel_size: int = 3,
      stride: int = 1,
      beta: float = 1.0,
      alpha: float = 0.2,
      activation: common.Activation = common.Activation.RELU,
      skipinit_gain: hk.initializers.Initializer = jnp.zeros,
      stochdepth_rate: Optional[float] = None,
      use_se: bool = False,
      se_ratio: float = 0.25,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.kernel_size = kernel_size
    self.activation = activation
    self.beta, self.alpha = beta, alpha
    self.skipinit_gain = skipinit_gain
    self.use_se, self.se_ratio = use_se, se_ratio
    # Bottleneck width.
    self.width = int(self.out_ch * bottleneck_ratio)
    self.stride = stride
    # Conv 0 (typically expansion conv).
    self.conv0 = common.WSConv2D(
        self.width, kernel_shape=1, padding='SAME', name='conv0')
    # Grouped NxN conv.
    self.conv1 = common.WSConv2D(
        self.width,
        kernel_shape=kernel_size,
        stride=stride,
        padding='SAME',
        name='conv1',
    )
    # Conv 2, typically projection conv.
    self.conv2 = common.WSConv2D(
        self.out_ch, kernel_shape=1, padding='SAME', name='conv2')
    # Use shortcut conv on channel change or downsample.
    self.use_projection = stride > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      self.conv_shortcut = common.WSConv2D(
          self.out_ch,
          kernel_shape=1,
          stride=stride,
          padding='SAME',
          name='conv_shortcut')
    # Are we using stochastic depth?
    self._has_stochdepth = (
        stochdepth_rate is not None and 0. < stochdepth_rate < 1.0)
    if self._has_stochdepth:
      self.stoch_depth = common.StochDepth(stochdepth_rate)

    if self.use_se:
      self.se = common.SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

  def __call__(
      self,
      x: jax.Array,
      is_training: bool,
  ) -> tuple[jax.Array, jax.Array]:
    """Applies the forward pass."""
    out = self.activation.fn(x) * self.beta
    shortcut = x
    if self.use_projection:  # Downsample with conv1x1.
      shortcut = self.conv_shortcut(out)
    out = self.conv0(out)
    out = self.conv1(self.activation.fn(out))
    out = self.conv2(self.activation.fn(out))
    if self.use_se:
      out = 2 * self.se(out) * out
    # Get average residual standard deviation for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    # Apply stochdepth if applicable.
    if self._has_stochdepth:
      out = self.stoch_depth(out, is_training)
    # Apply the kipInit Gain.
    out = out * hk.get_parameter(
        'skip_gain', (), out.dtype, init=self.skipinit_gain)
    return out * self.alpha + shortcut, res_avg_var
