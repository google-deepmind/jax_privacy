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

"""Import the NfNet models from the original codebase."""

import dataclasses

import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.experiments.image_classification.models import base
from jax_privacy.experiments.image_classification.models import common


nfnet_params = {
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 192, 'test_imsize': 256,
        'drop_rate': 0.2, 'RA_level': '405',},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 224, 'test_imsize': 320,
        'drop_rate': 0.3, 'RA_level': '410',},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'expansion': [0.5] * 4,
        'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 256, 'test_imsize': 352,
        'drop_rate': 0.4, 'RA_level': '410',},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 320, 'test_imsize': 416,
        'drop_rate': 0.4, 'RA_level': '415',},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 384, 'test_imsize': 512,
        'drop_rate': 0.5, 'RA_level': '415',},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 416, 'test_imsize': 544,
        'drop_rate': 0.5, 'RA_level': '415',},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 448, 'test_imsize': 576,
        'drop_rate': 0.5, 'RA_level': '415',},

    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'expansion': [0.5] * 4, 'group_width': [128] * 4,
        'big_width': [True] * 4,
        'train_imsize': 448, 'test_imsize': 576,
        'drop_rate': 0.5, 'RA_level': '415',},
}


def nfnet_config(
    variant: str,
    *,
    stochdepth_rate: float = 0.0,
    drop_rate: float | None = 0.0,
    restore_path: str | None = None,
) -> base.ModelConfig:
  """Creates a config for an NFNet.

  Args:
    variant: Variant of the NFNet model to use.
    stochdepth_rate: rate for stochastic depth.
    drop_rate: dropout-rate. If None, this *does not* deactivate it, but rather
      re-uses a default value specified per model.
    restore_path: CNS path to the model to restore.
  Returns:
    Model configuration.
  """
  return base.WithRestoreModelConfig(
      path=restore_path,
      params_key='params',
      network_state_key='state',
      layer_to_ignore='NFNet/~/linear',
      model=NFNetConfig(
          variant=variant,
          stochdepth_rate=stochdepth_rate,
          drop_rate=drop_rate,
      ),
  )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NFNetConfig(base.ModelConfig):
  """Configuration of NF-Net."""

  variant: str = 'F0'
  width: float = 1.0
  se_ratio: float = 0.5
  alpha: float = 0.2
  stochdepth_rate: float = 0.1
  drop_rate: float | None = None
  activation: common.Activation = common.Activation.SCALED_GELU
  fc_init: hk.initializers.Initializer | None = None
  # Multiplier for the final conv channel count
  final_conv_mult: int = 2
  final_conv_ch: int | None = None
  use_two_convs: bool = True
  name: str = 'NFNet'

  def make(self, num_classes: int) -> base.Model:
    return base.Model.from_hk_module(
        NFNet,
        num_classes=num_classes,
        variant=self.variant,
        width=self.width,
        se_ratio=self.se_ratio,
        alpha=self.alpha,
        stochdepth_rate=self.stochdepth_rate,
        drop_rate=self.drop_rate,
        activation=self.activation,
        fc_init=self.fc_init,
        final_conv_mult=self.final_conv_mult,
        final_conv_ch=self.final_conv_ch,
        use_two_convs=self.use_two_convs,
        name=self.name,
    )


class NFNet(hk.Module):
  """Normalizer-Free networks, The Next Generation."""

  variant_dict = nfnet_params

  def __init__(self, num_classes, variant='F0',
               width=1.0, se_ratio=0.5,
               alpha=0.2, stochdepth_rate=0.1, drop_rate=None,
               activation: common.Activation = common.Activation.SCALED_GELU,
               fc_init=None,
               # Multiplier for the final conv channel count
               final_conv_mult=2, final_conv_ch=None,
               use_two_convs=True,
               name='NFNet'):
    super().__init__(name=name)
    self.num_classes = num_classes
    self.variant = variant
    self.width = width
    self.se_ratio = se_ratio
    # Get variant info
    block_params = self.variant_dict[self.variant]
    self.train_imsize = block_params['train_imsize']
    self.test_imsize = block_params['test_imsize']
    self.width_pattern = block_params['width']
    self.depth_pattern = block_params['depth']
    self.bneck_pattern = block_params['expansion']
    self.group_pattern = block_params['group_width']
    self.big_pattern = block_params['big_width']
    self.activation = activation.fn
    if drop_rate is None:
      self.drop_rate = block_params['drop_rate']
    else:
      self.drop_rate = drop_rate
    self.which_conv = common.WSConv2D
    # Stem
    ch = self.width_pattern[0] // 2
    self.stem = hk.Sequential([
        self.which_conv(16, kernel_shape=3, stride=2,
                        padding='SAME', name='stem_conv0'),
        self.activation,
        self.which_conv(32, kernel_shape=3, stride=1,
                        padding='SAME', name='stem_conv1'),
        self.activation,
        self.which_conv(64, kernel_shape=3, stride=1,
                        padding='SAME', name='stem_conv2'),
        self.activation,
        self.which_conv(ch, kernel_shape=3, stride=2,
                        padding='SAME', name='stem_conv3'),
    ])

    # Body
    self.blocks = []
    expected_std = 1.0
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    stride_pattern = [1] + [2] * 3
    block_args = zip(self.width_pattern, self.depth_pattern, self.bneck_pattern,
                     self.group_pattern, self.big_pattern, stride_pattern)
    for (block_width, stage_depth, expand_ratio,
         group_size, big_width, stride) in block_args:
      for block_index in range(stage_depth):
        # Scalar pre-multiplier so each block sees an N(0,1) input at init
        beta = 1.0 / expected_std
        # Block stochastic depth drop-rate
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        out_ch = int(block_width * self.width)
        self.blocks += [NFBlock(ch, out_ch,
                                expansion=expand_ratio, se_ratio=se_ratio,
                                group_size=group_size,
                                stride=stride if block_index == 0 else 1,
                                beta=beta, alpha=alpha,
                                activation=self.activation,
                                which_conv=self.which_conv,
                                stochdepth_rate=block_stochdepth_rate,
                                big_width=big_width,
                                use_two_convs=use_two_convs,
                                )]
        ch = out_ch
        index += 1
         # Reset expected std but still give it 1 block of growth
        if block_index == 0:
          expected_std = 1.0
        expected_std = (expected_std**2 + alpha**2)**0.5

    # Head
    if final_conv_mult is None:
      if final_conv_ch is None:
        raise ValueError('Must provide one of final_conv_mult or final_conv_ch')
      ch = final_conv_ch
    else:
      ch = int(final_conv_mult * ch)
    self.final_conv = self.which_conv(ch, kernel_shape=1,
                                      padding='SAME', name='final_conv')
    if self.num_classes is not None:
      # By default, initialize with N(0, 0.01)
      if fc_init is None:
        fc_init = hk.initializers.RandomNormal(0.01, 0)
      self.fc = hk.Linear(self.num_classes, w_init=fc_init, with_bias=True)
    else:
      self.fc = None

  def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
    """Return the output of the final layer without any [log-]softmax."""
    # Stem
    out = self.stem(x)
    # Blocks
    for block in self.blocks:
      out, unused_res_avg_var = block(out, is_training=is_training)
    # Final-conv->activation, pool, dropout, classify
    out = self.activation(self.final_conv(out))
    pool = jnp.mean(out, [1, 2])
    if self.drop_rate > 0.0 and is_training:
      pool = hk.dropout(hk.next_rng_key(), self.drop_rate, pool)
    logits = self.fc(pool)
    return logits


class NFBlock(hk.Module):
  """Normalizer-Free RegNet Block."""

  def __init__(self, in_ch, out_ch, expansion=0.5, se_ratio=0.5,
               kernel_shape=3, group_size=128, stride=1,
               beta=1.0, alpha=0.2,
               which_conv=common.WSConv2D, activation=jax.nn.gelu,
               big_width=True, use_two_convs=True,
               stochdepth_rate=None, name=None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.expansion = expansion
    self.se_ratio = se_ratio
    self.kernel_shape = kernel_shape
    self.activation = activation
    self.beta, self.alpha = beta, alpha
    # Mimic resnet style bigwidth scaling?
    width = int((self.out_ch if big_width else self.in_ch) * expansion)
    # Round expanded with based on group count
    self.groups = width // group_size
    self.width = group_size * self.groups
    self.stride = stride
    self.use_two_convs = use_two_convs
    # Conv 0 (typically expansion conv)
    self.conv0 = which_conv(self.width, kernel_shape=1, padding='SAME',
                            name='conv0')
    # Grouped NxN conv
    self.conv1 = which_conv(self.width, kernel_shape=kernel_shape,
                            stride=stride, padding='SAME',
                            feature_group_count=self.groups, name='conv1')
    if self.use_two_convs:
      self.conv1b = which_conv(self.width, kernel_shape=kernel_shape,
                               stride=1, padding='SAME',
                               feature_group_count=self.groups, name='conv1b')
    # Conv 2, typically projection conv
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, padding='SAME',
                            name='conv2')
    # Use shortcut conv on channel change or downsample.
    self.use_projection = stride > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      self.conv_shortcut = which_conv(self.out_ch, kernel_shape=1,
                                      padding='SAME', name='conv_shortcut')
    # Squeeze + Excite Module
    self.se = common.SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

    # Are we using stochastic depth?
    self._has_stochdepth = (
        stochdepth_rate is not None and 1.0 > stochdepth_rate > 0.0)
    if self._has_stochdepth:
      self.stoch_depth = common.StochDepth(stochdepth_rate)

  def __call__(self, x, is_training):
    out = self.activation(x) * self.beta
    if self.stride > 1:  # Average-pool downsample.
      shortcut = hk.avg_pool(out, window_shape=(1, 2, 2, 1),
                             strides=(1, 2, 2, 1), padding='SAME')
      if self.use_projection:
        shortcut = self.conv_shortcut(shortcut)
    elif self.use_projection:
      shortcut = self.conv_shortcut(out)
    else:
      shortcut = x
    out = self.conv0(out)
    out = self.conv1(self.activation(out))
    if self.use_two_convs:
      out = self.conv1b(self.activation(out))
    out = self.conv2(self.activation(out))
    out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
    # Get average residual standard deviation for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    # Apply stochdepth if applicable.
    if self._has_stochdepth:
      out = self.stoch_depth(out, is_training)
    # SkipInit Gain
    out = out * hk.get_parameter('skip_gain', (), out.dtype, init=jnp.zeros)
    return out * self.alpha + shortcut, res_avg_var
