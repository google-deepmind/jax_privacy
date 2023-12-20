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

"""Common functions used to define model architectures."""

import enum
import os
from typing import Callable
import urllib.request

import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import utils
import numpy as np
import orbax.checkpoint as orbax_checkpoint


def _scaled(
    activation_fn: Callable[[jax.Array], jax.Array],
    scale: float,
) -> Callable[[jax.Array], jax.Array]:
  """Returns a scaled version of the activation function."""
  return lambda x: scale * activation_fn(x)


class Activation(enum.Enum):
  """Activations with and without scaling."""

  # Non-scaled activations.
  IDENTITY = enum.auto()
  CELU = enum.auto()
  ELU = enum.auto()
  GELU = enum.auto()
  GLU = enum.auto()
  LEAKY_RELU = enum.auto()
  LOG_SIGMOID = enum.auto()
  LOG_SOFTMAX = enum.auto()
  RELU = enum.auto()
  RELU6 = enum.auto()
  SELU = enum.auto()
  SIGMOID = enum.auto()
  SILU = enum.auto()
  SWISH = enum.auto()
  SOFT_SIGN = enum.auto()
  SOFTPLUS = enum.auto()
  TANH = enum.auto()

  # Scaled activations.
  SCALED_CELU = enum.auto()
  SCALED_ELU = enum.auto()
  SCALED_GELU = enum.auto()
  SCALED_GLU = enum.auto()
  SCALED_LEAKY_RELU = enum.auto()
  SCALED_LOG_SIGMOID = enum.auto()
  SCALED_LOG_SOFTMAX = enum.auto()
  SCALED_RELU = enum.auto()
  SCALED_RELU6 = enum.auto()
  SCALED_SELU = enum.auto()
  SCALED_SIGMOID = enum.auto()
  SCALED_SILU = enum.auto()
  SCALED_SWISH = enum.auto()
  SCALED_SOFT_SIGN = enum.auto()
  SCALED_SOFTPLUS = enum.auto()
  SCALED_TANH = enum.auto()

  @property
  def fn(self) -> Callable[[jax.Array], jax.Array]:
    match self:
      # Non-scaled activations.
      case Activation.IDENTITY:
        return jnp.asarray
      case Activation.CELU:
        return jax.nn.celu
      case Activation.ELU:
        return jax.nn.elu
      case Activation.GELU:
        return jax.nn.gelu
      case Activation.GLU:
        return jax.nn.glu
      case Activation.LEAKY_RELU:
        return jax.nn.leaky_relu
      case Activation.LOG_SIGMOID:
        return jax.nn.log_sigmoid
      case Activation.LOG_SOFTMAX:
        return jax.nn.log_softmax
      case Activation.RELU:
        return jax.nn.relu
      case Activation.RELU6:
        return jax.nn.relu6
      case Activation.SELU:
        return jax.nn.selu
      case Activation.SIGMOID:
        return jax.nn.sigmoid
      case Activation.SILU:
        return jax.nn.silu
      case Activation.SWISH:
        return jax.nn.silu
      case Activation.SOFT_SIGN:
        return jax.nn.soft_sign
      case Activation.SOFTPLUS:
        return jax.nn.softplus
      case Activation.TANH:
        return jnp.tanh
      # Scaled activations.
      case Activation.SCALED_CELU:
        return _scaled(Activation.CELU.fn, 1.270926833152771)
      case Activation.SCALED_ELU:
        return _scaled(Activation.ELU.fn, 1.2716004848480225)
      case Activation.SCALED_GELU:
        return _scaled(Activation.GELU.fn, 1.7015043497085571)
      case Activation.SCALED_GLU:
        return _scaled(Activation.GLU.fn, 1.8484294414520264)
      case Activation.SCALED_LEAKY_RELU:
        return _scaled(Activation.LEAKY_RELU.fn, 1.70590341091156)
      case Activation.SCALED_LOG_SIGMOID:
        return _scaled(Activation.LOG_SIGMOID.fn, 1.9193484783172607)
      case Activation.SCALED_LOG_SOFTMAX:
        return _scaled(Activation.LOG_SOFTMAX.fn, 1.0002083778381348)
      case Activation.SCALED_RELU:
        return _scaled(Activation.RELU.fn, 1.7139588594436646)
      case Activation.SCALED_RELU6:
        return _scaled(Activation.RELU6.fn, 1.7131484746932983)
      case Activation.SCALED_SELU:
        return _scaled(Activation.SELU.fn, 1.0008515119552612)
      case Activation.SCALED_SIGMOID:
        return _scaled(Activation.SIGMOID.fn, 4.803835391998291)
      case Activation.SCALED_SILU:
        return _scaled(Activation.SILU.fn, 1.7881293296813965)
      case Activation.SCALED_SWISH:
        return _scaled(Activation.SWISH.fn, 1.7881293296813965)
      case Activation.SCALED_SOFT_SIGN:
        return _scaled(Activation.SOFT_SIGN.fn, 2.338853120803833)
      case Activation.SCALED_SOFTPLUS:
        return _scaled(Activation.SOFTPLUS.fn, 1.9203323125839233)
      case Activation.SCALED_TANH:
        return _scaled(Activation.TANH.fn, 1.5939117670059204)


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

  def __call__(self, inputs: jax.Array, eps: float = 1e-4) -> jax.Array:
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
      name: str | None = None,
  ):
    super().__init__(name=name)
    self.drop_rate = drop_rate
    self.scale_by_keep = scale_by_keep

  def __call__(self, x: jax.Array, is_training: bool) -> jax.Array:
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
      hidden_ch: int | None = None,
      activation: Activation = Activation.RELU,
      name: str | None = None,
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
    h = self.fc1(self.activation.fn(self.fc0(h)))
    # Broadcast along H, W dimensions.
    h = jax.nn.sigmoid(h)[:, None, None]
    return h


def download_and_get_filename(
    name: str,
    root_dir: str = '/tmp/jax_privacy',
) -> str:
  """Load file, downloading to /tmp/jax_privacy first if necessary."""
  local_path = os.path.join(root_dir, name)
  if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path))
  if not os.path.exists(local_path):
    gcp_bucket_url = 'gs://dm_jax_privacy/models/'
    download_url = gcp_bucket_url + name
    urllib.request.urlretrieve(download_url, local_path)
  return local_path


def restore_from_path(
    *,
    restore_path: str,
    params_key: str,
    network_state_key: str,
    layer_to_reset: str | None,
    params_init: hk.Params,
    network_state_init: hk.State,
) -> tuple[hk.Params, hk.State]:
  """Restore parameters and model state from an existing checkpoint.

  Args:
    restore_path: path to model to restore. This should point to a dict that can
      be loaded through orbax.
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
  full_path = download_and_get_filename(restore_path)

  ckpt_state = orbax_checkpoint.PyTreeCheckpointer().restore(full_path)
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
