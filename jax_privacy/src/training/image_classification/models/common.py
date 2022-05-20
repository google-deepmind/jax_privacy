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

"""Common functions used to define model architectures."""
import os
from typing import Callable, Optional, Tuple
import urllib.request

import chex
import dill
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import utils
import numpy as np


Activation = Callable[[chex.Array], chex.Array]

# Activations with and without scaling.
activations_dict = {
    # Regular activations.
    'identity': lambda x: x,
    'celu': jax.nn.celu,
    'elu': jax.nn.elu,
    'gelu': jax.nn.gelu,
    'glu': jax.nn.glu,
    'leaky_relu': jax.nn.leaky_relu,
    'log_sigmoid': jax.nn.log_sigmoid,
    'log_softmax': jax.nn.log_softmax,
    'relu': jax.nn.relu,
    'relu6': jax.nn.relu6,
    'selu': jax.nn.selu,
    'sigmoid': jax.nn.sigmoid,
    'silu': jax.nn.silu,
    'swish': jax.nn.silu,
    'soft_sign': jax.nn.soft_sign,
    'softplus': jax.nn.softplus,
    'tanh': jnp.tanh,

    # Scaled activations.
    'scaled_celu': lambda x: jax.nn.celu(x) * 1.270926833152771,
    'scaled_elu': lambda x: jax.nn.elu(x) * 1.2716004848480225,
    'scaled_gelu': lambda x: jax.nn.gelu(x) * 1.7015043497085571,
    'scaled_glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'scaled_leaky_relu': lambda x: jax.nn.leaky_relu(x) * 1.70590341091156,
    'scaled_log_sigmoid': lambda x: jax.nn.log_sigmoid(x) * 1.9193484783172607,
    'scaled_log_softmax': lambda x: jax.nn.log_softmax(x) * 1.0002083778381348,
    'scaled_relu': lambda x: jax.nn.relu(x) * 1.7139588594436646,
    'scaled_relu6': lambda x: jax.nn.relu6(x) * 1.7131484746932983,
    'scaled_selu': lambda x: jax.nn.selu(x) * 1.0008515119552612,
    'scaled_sigmoid': lambda x: jax.nn.sigmoid(x) * 4.803835391998291,
    'scaled_silu': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'scaled_swish': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'scaled_soft_sign': lambda x: jax.nn.soft_sign(x) * 2.338853120803833,
    'scaled_softplus': lambda x: jax.nn.softplus(x) * 1.9203323125839233,
    'scaled_tanh': lambda x: jnp.tanh(x) * 1.5939117670059204,
}


class WSConv2D(hk.Conv2D):
  """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

  @hk.transparent
  def standardize_weight(self, weight, eps=1e-4):
    """Apply scaled WS with affine gain."""
    mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
    fan_in = np.prod(weight.shape[:-1])
    gain = hk.get_parameter('gain', shape=(weight.shape[-1],),
                            dtype=weight.dtype, init=jnp.ones)
    # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var).
    scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
    shift = mean * scale
    return weight * scale - shift

  def __call__(self, inputs: chex.Array, eps: float = 1e-4) -> chex.Array:
    w_shape = self.kernel_shape + (
        inputs.shape[self.channel_index] // self.feature_group_count,
        self.output_channels)
    # Use fan-in scaled init, but WS is largely insensitive to this choice.
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'normal')
    w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
    weight = self.standardize_weight(w, eps)
    out = jax.lax.conv_general_dilated(
        inputs, weight, window_strides=self.stride, padding=self.padding,
        lhs_dilation=self.lhs_dilation, rhs_dilation=self.kernel_dilation,
        dimension_numbers=self.dimension_numbers,
        feature_group_count=self.feature_group_count)
    # Always add bias.
    bias_shape = (self.output_channels,)
    bias = hk.get_parameter('bias', bias_shape, inputs.dtype, init=jnp.zeros)
    return out + bias


class StochDepth(hk.Module):
  """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

  def __init__(
      self,
      drop_rate: float,
      scale_by_keep: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.drop_rate = drop_rate
    self.scale_by_keep = scale_by_keep

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    if not is_training:
      return x
    batch_size = x.shape[0]
    r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1],
                           dtype=x.dtype)
    keep_prob = 1. - self.drop_rate
    binary_tensor = jnp.floor(keep_prob + r)
    if self.scale_by_keep:
      x = x / keep_prob
    return x * binary_tensor


class SqueezeExcite(hk.Module):
  """Simple Squeeze+Excite module."""

  def __init__(
      self,
      in_ch: int,
      out_ch: int,
      se_ratio: float = 0.5,
      hidden_ch: Optional[int] = None,
      activation: Activation = jax.nn.relu,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    if se_ratio is None:
      if hidden_ch is None:
        raise ValueError('Must provide one of se_ratio or hidden_ch')
      self.hidden_ch = hidden_ch
    else:
      self.hidden_ch = max(1, int(self.in_ch * se_ratio))
    self.activation = activation
    self.fc0 = hk.Linear(self.hidden_ch, with_bias=True)
    self.fc1 = hk.Linear(self.out_ch, with_bias=True)

  def __call__(self, x):
    # Average over HW dimensions.
    h = jnp.mean(x, axis=[1, 2])
    h = self.fc1(self.activation(self.fc0(h)))
    # Broadcast along H, W dimensions.
    h = jax.nn.sigmoid(h)[:, None, None]
    return h


def download_and_read_file(name: str, root_dir: str = '/tmp/jax_privacy'):
  """Load file, downloading to /tmp/jax_privacy first if necessary."""
  local_path = os.path.join(root_dir, name)
  if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path))
  if not os.path.exists(local_path):
    gcp_bucket_url = 'https://storage.googleapis.com/dm_jax_privacy/models/'
    download_url = gcp_bucket_url + name
    urllib.request.urlretrieve(download_url, local_path)
  return open(local_path, mode='rb')


def restore_from_path(
    *,
    restore_path: str,
    params_key: str,
    network_state_key: str,
    layer_to_reset: Optional[str],
    params_init: chex.ArrayTree,
    network_state_init: chex.ArrayTree,
) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
  """Restore parameters and model state from an existing checkpoint.

  Args:
    restore_path: path to model to restore. This should point to a dict that can
      be loaded through dill.
    params_key: key of the dict corresponding to the model parameters.
    network_state_key: key of the dict corresponding to the model state.
    layer_to_reset: name of the layer to reset (exact match required).
    params_init: initial value for the model parameters (used only if
      a layer matches with `layer_to_reset`).
    network_state_init: initial value for the model state (used only if
      a layer matches with `layer_to_reset`).
  Returns:
    params: model parameters loaded from the checkpoint (with the classifier
      potentially reset).
    network_state: model state loaded from the checkpoint (with state associated
      to the classifier potentially reset).
  """
  # Load pretrained experiment state.
  with download_and_read_file(restore_path) as f:
    ckpt_state = dill.load(f)

  params_loaded = utils.bcast_local_devices(ckpt_state[params_key])
  network_state_loaded = utils.bcast_local_devices(
      ckpt_state[network_state_key])

  def should_reset_layer(module_name, *_):
    return module_name == layer_to_reset

  if layer_to_reset:
    _, params_loaded = hk.data_structures.partition(
        should_reset_layer, params_loaded)
    _, network_state_loaded = hk.data_structures.partition(
        should_reset_layer, network_state_loaded)

  # Note that the 'loaded' version must be last in the merge to get priority.
  params = hk.data_structures.merge(params_init, params_loaded)
  network_state = hk.data_structures.merge(
      network_state_init, network_state_loaded)

  return params, network_state
