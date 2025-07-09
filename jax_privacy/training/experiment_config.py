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

"""Experiment configuration."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Literal, Protocol

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy import accounting
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import typing
from jax_privacy.training import algorithm_config
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
    grad_clipping: Whether to log the proportion of per-example gradients that
      get clipped at each iteration.
    snr_global: Whether to log the Signal-to-Noise Ratio (SNR) globally across
      layers, where the SNR is defined as: ||non_private_grads||_2 /
      ||noise||_2.
    snr_per_layer: Whether to log the Signal-to-Noise Ratio (SNR) per layer,
      where the SNR is defined as: ||non_private_grads||_2 / ||noise||_2.
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
          '%s total_size=%i', prefix, hk.data_structures.tree_size(params)
      )
      for layer_name in params:
        layer = params[layer_name]
        logging.info(
            '%s%s size=%i shapes=%s',
            prefix,
            layer_name,
            hk.data_structures.tree_size(layer),
            jax.tree_util.tree_map(jnp.shape, layer),
        )

  def additional_training_metrics(
      self, params: hk.Params, grads: hk.Params
  ) -> Mapping[str, chex.Numeric]:
    """Returns additional metrics based on `params` and `grads`.

    Might be used, for example, to log metrics during privacy auditing.

    Args:
      params: Model parameters.
      grads: Gradients.
    """
    del params, grads
    return {}


@dataclasses.dataclass(kw_only=True, slots=True)
class DpConfig:
  """Configuration to activate / deactivate DP training.

  Attributes:
    clipping_norm: maximal l2 norm to clip each gradient per sample. No clipping
      is applied if it is set to either `None` or `float('inf')`.
    rescale_to_unit_norm: If true, additionally rescale the clipped gradient by
      1/clipping_norm so it has an L2 norm of at most one.
    per_example_grad_method: Per-example gradient clipping method to use. Does
      not affect the results, but controls speed/memory trade-off. Unrolling
      with a `jax.lax.scan` loop is usually faster when the program requires a
      large amount of device memory (e.g. large batch sizes per device);
      otherwise, vectorization is faster.
    auto_tune_target_epsilon: DP epsilon to use for auto-tuning.
    algorithm: Specifies the algorithm to use, be it Non-Private, DP-SGD, or
      DP-FTRL.
    analysis_method: defines how to analyze the private algorithm, e.g., using
      standard dp-sgd Poisson-amplified account or as a single-release. The
      caller must ensure this analysis is applicable to the algorithm being run,
      otherwise, an error will be raised.
    sampling_method: if our privacy analysis assumes sampling, which sampling
      method it should assume. See SamplingMethod enum for details on each
      sampling method and the adjacency definitions it assumes.
    delta: DP delta to use to compute DP guarantees.
    auto_tune_field: whether to automatically adapt a hyper-parameter to fit the
      privacy budget. It should be set to one of 'batch_size',
      'noise_multiplier', 'auto_tune_target_epsilon', 'num_updates', or None.
    dp_accountant: Configuration for the DP accountant to use.
    dp_guarantee_is_finite: Whether the configuration is compatible with finite
      DP guarantees.
    accountant_cache_num_points: Number of points to pre-compute and cache for
      the accountant.
  """

  # Gradient clipping options.
  clipping_norm: float | None
  rescale_to_unit_norm: bool = True
  per_example_grad_method: grad_clipping.PerExampleGradMethod = (
      grad_clipping.UNROLLED
  )
  # Accounting options.
  delta: float
  # TODO: b/415360727 - If only DpSgdConfig is being used here, consider
  # replacing with DpSgdConfig.
  algorithm: algorithm_config.AlgorithmConfig
  analysis_method: Literal['DP-SGD', 'Single-Release'] = 'DP-SGD'
  sampling_method: accounting.SamplingMethod = accounting.SamplingMethod.POISSON
  dp_accountant: accounting.DpAccountantConfig = dataclasses.field(
      default_factory=accounting.PldAccountantConfig
  )
  # Auto-tuning options.
  auto_tune_field: typing.AutoTuneField = None
  auto_tune_target_epsilon: float | None = None
  # Caching options.
  accountant_cache_num_points: int = 100

  @classmethod
  def deactivated(cls) -> 'DpConfig':
    return DpConfig(
        delta=1.0,
        clipping_norm=None,
        rescale_to_unit_norm=False,
        algorithm=algorithm_config.NoDpConfig(),
    )

  def __post_init__(self):
    if self.clipping_norm is None:
      self.clipping_norm = float('inf')
    self._validate()

  def _validate(self):
    if self.clipping_norm < 0:
      raise ValueError('Clipping norm must be non-negative.')
    elif not self.clipping_norm and self.rescale_to_unit_norm:
      raise ValueError('Rescaling to unit norm without clipping.')

  @property
  def dp_guarantee_is_finite(self) -> bool:
    nonzero_noise = bool(self.algorithm.noise_multiplier)
    bounded_norm = bool(np.isfinite(self.clipping_norm))
    return nonzero_noise and bounded_norm

  def make_accountant(
      self,
      *,
      num_samples: int,
      batch_size: int,
      batch_size_scale_schedule: accounting.BatchingScaleSchedule | None = None,
      examples_per_user: int | None = None,
      cycle_length: int | None = None,
      truncated_batch_size: int | None = None,
  ) -> tuple[accounting.DpTrainingAccountant, accounting.DpParams]:
    """Creates the accountant for the experiment."""
    params = accounting.DpParams(
        noise_multipliers=self.algorithm.noise_multiplier,
        delta=self.delta,
        num_samples=num_samples,
        batch_size=batch_size,
        batch_size_scale_schedule=batch_size_scale_schedule,
        examples_per_user=examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
        sampling_method=self.sampling_method,
        is_finite_guarantee=self.dp_guarantee_is_finite,
    )
    match self.analysis_method:
      case 'DP-SGD':
        if examples_per_user is None or examples_per_user == 1:
          accountant = accounting.DpsgdTrainingAccountant(
              dp_accountant_config=self.dp_accountant
          )
        else:
          accountant = accounting.DpsgdTrainingUserLevelAccountant(
              dp_accountant_config=self.dp_accountant
          )
      case 'Single-Release':
        accountant = accounting.SingleReleaseTrainingAccountant(
            dp_accountant_config=self.dp_accountant
        )
    return accountant, params


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
      specified as a string, train only layer whose name is an exact match of
      this string. If specified as a filter function, it will be called on each
      `(layer_name, parameter_name parameter_value)` to determine whether the
      parameter should be trained.
  """

  num_updates: int
  batch_size: BatchSizeTrainConfig
  dp: DpConfig
  logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
  weight_decay: float = 0.0
  train_only_layer: str | FilterFn | None = None

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
