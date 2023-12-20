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

"""Experiment configuration."""

import dataclasses
from typing import Any, Optional, Protocol, Union

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.src import accounting
from jax_privacy.src.dp_sgd import typing
import numpy as np


class FilterFn(Protocol):

  def __call__(
      self,
      module_name: str,
      parameter_name: str,
      parameter_value: Any,
  ) -> bool:
    """Predicate function compatible with `haiku.data_structures` functions.

    Args:
      module_name: name of the haiku layer.
      parameter_name: name of the haiku parameter (within the layer).
      parameter_value: value of the parameter.
    """


@dataclasses.dataclass(kw_only=True, slots=True)
class LoggingConfig:
  """Logging configuration.

  Attributes:
    grad_clipping: Whether to log the proportion of per-example gradients
        that get clipped at each iteration.
    snr_global: Whether to log the Signal-to-Noise Ratio (SNR) globally
        across layers, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
    snr_per_layer: Whether to log the Signal-to-Noise Ratio (SNR) per
        layer, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
    prepend_split_name: Whether to prepend the name of the split to metrics
      being logged (e.g. 'train/loss' instead of 'loss').
    log_params_shapes: Whether to log parameter shapes (called during
      compilation).
    grad_sparsity: Whether to log the number of non-"approximately zero"
      gradient coordinates.
    grad_sparsity_threshold: threshold to determine that a coordinate is
      approximately zero.
  """

  grad_clipping: bool = False
  snr_global: bool = False
  snr_per_layer: bool = False
  prepend_split_name: bool = False
  log_params_shapes: bool = True

  def maybe_log_param_shapes(self, params: hk.Params, prefix: str = ''):
    """Logs information about `params` if `log_params_shapes` is True."""
    if self.log_params_shapes:
      logging.info(
          '%s total_size=%i', prefix, hk.data_structures.tree_size(params))
      for layer_name in params:
        layer = params[layer_name]
        logging.info(
            '%s%s size=%i shapes=%s',
            prefix,
            layer_name,
            hk.data_structures.tree_size(layer),
            jax.tree_map(jnp.shape, layer),
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class DPConfig:
  """Configuration to activate / deactivate DP training.

  Attributes:
    clipping_norm: maximal l2 norm to clip each gradient per sample. No
      clipping is applied if it is set to either `None` or `float('inf')`.
    rescale_to_unit_norm: whether to rescale to an l2 norm of 1
      after each gradient per sample has been clipped to `clipping_norm`.
    vectorize_grad_clipping: whether the computation of gradients clipped
      per sample is to be vectorized across the mini batch. Otherwise, a
      (JAX) loop is used to iterate over the mini-batch. Using a `for` loop is
      usually faster when the program requires a large amount of device memory
      (e.g. large batch sizes per device), otherwise vectorization is faster.
    auto_tune_target_epsilon: DP epsilon to use for auto-tuning.
    noise_multiplier: noise multiplier to use in DP-SGD.
    delta: DP delta to use to compute DP guarantees.
    auto_tune_field: whether to automatically adapt a hyper-parameter to fit the
      privacy budget. It should be set to one of 'batch_size',
      'noise_multiplier', 'auto_tune_target_epsilon', 'num_updates', or None.
    accountant: Configuration for the DP accountant to use.
    dp_guarantee_is_finite: Whether the configuration is compatible with finite
      DP guarantees.
    accountant_cache_num_points: Number of points to pre-compute and cache for
      the accountant.
  """

  # Gradient clipping options.
  clipping_norm: Optional[float]
  rescale_to_unit_norm: bool = True
  vectorize_grad_clipping: bool = False
  # Accounting options.
  delta: float
  noise_multiplier: Optional[float]
  accountant: accounting.DpAccountantConfig = dataclasses.field(
      default_factory=accounting.PldAccountantConfig)
  # Auto-tuning options.
  auto_tune_field: typing.AutoTuneField = None
  auto_tune_target_epsilon: Optional[float] = None
  # Caching options.
  accountant_cache_num_points: int = 100

  @classmethod
  def deactivated(cls) -> 'DPConfig':
    return DPConfig(
        delta=1.0,
        clipping_norm=None,
        rescale_to_unit_norm=False,
        noise_multiplier=None,
    )

  def __post_init__(self):
    if self.clipping_norm is None:
      self.clipping_norm = float('inf')
    if self.noise_multiplier is None:
      self.noise_multiplier = 0.0
    self._validate()

  def _validate(self):
    if self.clipping_norm < 0:
      raise ValueError('Clipping norm must be non-negative.')
    elif not self.clipping_norm and self.rescale_to_unit_norm:
      raise ValueError('Rescaling to unit norm without clipping.')
    if self.noise_multiplier < 0:
      raise ValueError('Noise multiplier must be non-negative.')

  @property
  def dp_guarantee_is_finite(self) -> bool:
    return bool(self.noise_multiplier) and bool(np.isfinite(self.clipping_norm))

  def make_accountant(
      self,
      num_samples: int,
      batch_size: int,
      batch_size_scale_schedule: accounting.BatchingScaleSchedule = None,
  ) -> accounting.ExperimentAccountant:
    """Creates the accountant for the experiment."""
    return accounting.ExperimentAccountant(
        noise_multiplier=self.noise_multiplier,
        dp_delta=self.delta,
        num_samples=num_samples,
        batch_size=batch_size,
        batch_size_scale_schedule=batch_size_scale_schedule,
        dp_accountant_config=self.accountant,
        is_finite_guarantee=self.dp_guarantee_is_finite,
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class BatchSizeTrainConfig:
  """Configuration for the batch-size at training time.

  Attributes:
    total: total batch-size to use.
    per_device_per_step: batch-size to use on each device on each step. This
      number should divide `total` * `jax.device_count()`.
    scale_schedule: schedule for scaling the batch-size.
  """
  total: int
  per_device_per_step: int
  scale_schedule: accounting.BatchingScaleSchedule = None


@dataclasses.dataclass(kw_only=True, slots=True)
class TrainingConfig:
  """Configuration for training.

  Attributes:
    num_updates: Number of training updates.
    batch_size: batch size configuration.
    dp: DP configuration.
    logging: logging configuration.
    weight_decay: weight-decay to apply to the model weights during training.
    train_only_layer: if set to None, train all layers of the models. If
        specified as a string, train only layer whose name is an exact match
        of this string. If specified as a filter function, it will be
        called on each `(layer_name, parameter_name parameter_value)` to
        determine whether the parameter should be trained.
  """

  num_updates: int
  batch_size: BatchSizeTrainConfig
  dp: DPConfig
  logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
  weight_decay: float = 0.0
  train_only_layer: Optional[Union[str, FilterFn]] = None

  def is_trainable(self, module_name: str, param_name: str, param: Any) -> bool:
    if self.train_only_layer is None:
      return True
    elif isinstance(self.train_only_layer, str):
      return module_name == self.train_only_layer
    else:
      return self.train_only_layer(module_name, param_name, param)


@dataclasses.dataclass(kw_only=True, slots=True)
class EvaluationConfig:
  """Configuration for the evaluation.

  Attributes:
    batch_size: Batch-size for evaluation.
    max_num_batches: Maximum number of batches to use in an evaluation run.
  """

  batch_size: int
  max_num_batches: int | None = None
