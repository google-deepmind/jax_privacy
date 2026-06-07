# Copyright 2026 DeepMind Technologies Limited.
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

"""Module for defining DP Execution Plans.

**API Stability:**
- `DPExecutionPlan`: 9/10 -- Stable, backwards-compatible changes possible.
- `BandMFExecutionPlanConfig`: 7/10 -- Mostly stable, minor changes possible.

# Writing General-Purpose DP Training Loops via DPExecutionPlan

This module introduces the `DPExecutionPlan`, an object designed to encapsulate
the core components of a differentially private (DP) mechanism. The primary aim
is to simplify the process of constructing and applying DP mechanisms by
packaging these components cohesively. A key benefit is the assurance that, when
used correctly, the combination of these components will achieve the stated DP
properties.

The design is framework-agnostic, specifying the essential pillars of a DP
mechanism—such as batch selection and noise addition—without tightly coupling
them to a specific training loop. Each component is exposed through a simple,
well-documented API, allowing for flexible integration into various frameworks
or direct use with JAX.

By programming against our DPExecutionPlan interface, it is easy to swap out
different components or entire mechanisms without changing the core training
loop logic. While the components are designed to work together, they can also be
used selectively. For instance, a researcher might choose to use only the noise
addition component if their dataset doesn't support the efficient random access
required by the batch selection strategy. Though this might invalidate the
formal DP guarantee, it can still be valuable for research or when a heuristic
quantification of privacy is acceptable. Ultimately, `DPExecutionPlan` aims to
free users to concentrate on their training pipeline setup, rather than on the
intricacies of correctly assembling DP components to achieve a desired privacy
guarantee.

# Selecting and using a DPExecutionPlan

Constructors for these plans are highly configurable, offering access to the
full capabilities of the underlying components while also providing sensible
defaults. Our primary entry point is currently `BandMFExecutionPlanConfig`,
although more will become available in the future when we feel the API has
stabilized.
"""

import dataclasses
import functools
from typing import Callable

import dp_accounting
import jax
from jax_privacy import batch_selection
from jax_privacy import clipping
from jax_privacy import noise_addition
from jax_privacy.experimental import accounting
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax
import pydantic

NeighboringRelation = dp_accounting.NeighboringRelation
AccountantFn = Callable[[NeighboringRelation], dp_accounting.PrivacyAccountant]


@dataclasses.dataclass(frozen=True)
class PerformanceFlags:
  """Performance-only flags that do not affect mechanism behavior or privacy.

  These flags control implementation details such as numerical precision,
  sharding strategy, and memory/compute trade-offs. Changing them should
  not alter the privacy properties or the mathematical definition of the
  DP mechanism.

  Attributes:
    dtype: The dtype to use for noise generation and gradient aggregation.
    noise_seed: A seed for the random number generator used for noise addition.
    intermediate_strategy: Strategy for generating intermediate noise, controls
      sharding behavior for noise addition.
    microbatch_size: If set, per-example gradient computation is broken into
      sequential microbatches to reduce peak memory at the cost of compute.
    spmd_axis_name: Axis name for distributed vmap in SPMD settings.
  """

  dtype: jax.typing.DTypeLike = np.float32
  noise_seed: int | None = None
  intermediate_strategy: noise_addition.SupportedStrategies = (
      noise_addition.SupportedStrategies.DEFAULT
  )
  microbatch_size: int | None = None
  spmd_axis_name: str | None = None


@dataclasses.dataclass(frozen=True)
class DPExecutionPlan:
  """Class for defining a DP execution plan.

  A DP execution plan consists of a collection of components which when used
  together in the expected manner determine the DP guarantee, along with a
  DpEvent which precisely quantifies it. If constructed via one of the
  ExecutionPlanConfig classes defined in this module, then the `dp_event` can
  be trusted as having been formally verified by the JAX Privacy authors.

  In pseudo-code, the components of this dataclass should roughly be used as
  follows:

  .. code-block:: python

    plan = ... # Plan depending on the flavor of DP training you want
    noise_state = plan.noise_addition_transform.init(...)
    batch_sampler = plan.batch_selection_strategy
    for indices in batch_sampler.batch_iterator(num_examples):
      batch = data.select(indices)
      grad_fn = plan.clipped_grad(loss_fn)
      clipped_grad_sum = grad_fn(params, batch, ...)

      dp_grad, noise_state = plan.noise_addition_transform.update(
          clipped_grad_sum, noise_state
      )
      # Sensitive, discard immediately after use.
      del indices, batch, clipped_grad_sum
      # Arbitrary post-processing of dp_grad.

  If possible, we recommend coupling the batch selection, clipped aggregation,
  and noise addition components as tightly as possible to ensure sensitive
  objects are not intercepted and used unintentionally. For example, it is
  critical that no modification is applied to the `clipped_grad_sum` (such as
  scaling) before the noise_addition_transform is applied, as such a
  modifications could invalidate the DP guarantee because the noise is
  calibrated based on the sensivity of the clipped_grad_sum.

  Attributes:
    clipped_grad: A function with a similar signature to jax.value_and_grad, but
      computes a sum of per-example clipped gradients.
    batch_selection_strategy: Determines how batches are formed in each
      iteration.
    noise_addition_transform: Stateful transformation that adds noise to clipped
      and aggregated gradients after each iteration.
    dp_event: Characterizes the mechanism in terms of primitive building blocks
      that dp_accounting knows how to analyze.
    neighboring_relation: The DP neighboring relation assumed by this mechanism.
  """

  clipped_grad: Callable[..., clipping.BoundedSensitivityCallable]
  batch_selection_strategy: batch_selection.BatchSelectionStrategy
  noise_addition_transform: optax.GradientTransformation
  dp_event: dp_accounting.DpEvent
  neighboring_relation: NeighboringRelation


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(arbitrary_types_allowed=True),
)  # pytype: disable=wrong-keyword-args
class BandMFExecutionPlanConfig:
  """Configuration for an Amplified BandMF-based DPExecutionPlan.

  This config is designed to be fully serializable, defined in terms of simple
  types. The config can be created with or without a noise_multiplier. If
  created without one, call `calibrate()` to obtain a new config with a
  noise_multiplier calibrated to a target (epsilon, delta) guarantee.

  The expected batch size of the batch selection strategy is
  `(num_examples / num_bands) * sampling_prob`. `num_examples` may or may not be
  passed to this config. It should only be passed if it is considered a public,
  non-sensitive quantity (i.e., when using zero-out adjacency rather than
  add-remove).

  Example Usage (Calibrate from epsilon/delta):
    >>> config = BandMFExecutionPlanConfig.default(  # doctest: +SKIP
    ...   num_bands=1, iterations=1000, sampling_prob=0.1,
    ... ).calibrate(epsilon=1.0, delta=1e-5)

  Example Usage (Direct noise_multiplier):
    >>> config = BandMFExecutionPlanConfig.default(
    ...   num_bands=1, iterations=1000, sampling_prob=0.1,
    ...   noise_multiplier=1.0,
    ... )

  Example Usage (BandMF with custom strategy):
    >>> config = BandMFExecutionPlanConfig(  # doctest: +SKIP
    ...   strategy=np.array([1.0, 0.5, 0.2]),
    ...   iterations=1000, sampling_prob=0.4,
    ... ).calibrate(epsilon=1.0, delta=1e-5)

  References: https://arxiv.org/abs/2306.08153 and
  https://arxiv.org/abs/2405.15913

  Attributes:
    noise_multiplier: The ratio of noise standard deviation to the query
      sensitivity. The actual noise stddev is determined by this value, the
      query sensitivity, and the strategy matrix column norm. If not set, use
      `calibrate()` to automatically determine it from target (epsilon, delta)
      privacy parameters.
    iterations: The number of iterations the mechanism is defined for. Tip: Set
      this to be a multiple of num_bands for the best utility.
    strategy: The toeplitz coefficeints of the BandMF strategy matrix.
    l2_clip_norm: The maximum L2 norm of the per-example gradients.
    rescale_to_unit_norm: Divide the clipped gradient by the l2_clip_norm.
    normalize_by: Divide the sum-of-clipped gradients by this value.
    sampling_prob: The Poisson sampling probability for each example in a group.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to. If set, the plan.batch_selection_strategy will always
      return batches of size at most truncated_batch_size, and accounting will
      be based on truncated Poisson sampling (http://arxiv.org/html/2508.15089).
    num_examples: The number of examples in the dataset. Required when
      truncated_batch_size is set. Only set when the dataset size is considered
      public, non-sensitive information (e.g., when using zero-out adjacency
      rather than add-remove). If specified, the dataset will be partitioned
      using `batch_selection.PartitionType.EQUAL_SPLIT`, and otherwise it will
      be partitioned using `batch_selection.PartitionType.INDEPENDENT`.
    column_normalize: Whether to column-normalize the strategy matrix.
  """

  iterations: int = pydantic.Field(ge=0)
  strategy: np.typing.ArrayLike = pydantic.Field()
  noise_multiplier: float | None = pydantic.Field(default=None, ge=0)
  l2_clip_norm: float = pydantic.Field(default=1.0, ge=0)
  rescale_to_unit_norm: bool = pydantic.Field(default=True)
  normalize_by: float = pydantic.Field(default=1.0, ge=0)
  sampling_prob: float = pydantic.Field(default=1.0, ge=0, le=1)
  truncated_batch_size: int | None = pydantic.Field(default=None, ge=0)
  num_examples: int | None = pydantic.Field(default=None, ge=0)
  column_normalize: bool = pydantic.Field(default=False)

  def __post_init__(self):
    if self.truncated_batch_size is not None and self.num_examples is None:
      raise ValueError('truncated_batch_size requires num_examples to be set.')
    strategy = np.asarray(self.strategy)
    if strategy.ndim != 1:
      raise ValueError(f'strategy must be a 1D array, found {strategy.ndim}.')
    if strategy.size == 0 or strategy.size > self.iterations:
      raise ValueError(f'{strategy.size=} not in range [1, {self.iterations}].')

  @property
  def _neighboring_relation(self) -> NeighboringRelation:
    """Returns the neighboring relation and partition type for the config."""
    if self.num_examples is not None:
      return NeighboringRelation.REPLACE_SPECIAL
    return NeighboringRelation.ADD_OR_REMOVE_ONE

  @property
  def _partition_type(self) -> batch_selection.PartitionType:
    """Returns the partition type for the config."""
    if self.num_examples is not None:
      return batch_selection.PartitionType.EQUAL_SPLIT
    return batch_selection.PartitionType.INDEPENDENT

  def _get_dp_event(self, sigma: float) -> dp_accounting.DpEvent:
    """Returns a DpEvent for the BandMF mechanism."""
    num_bands = len(self.strategy)
    if self.truncated_batch_size:
      group_size = self.num_examples // num_bands
      return accounting.truncated_amplified_bandmf_event(
          noise_multiplier=sigma,
          iterations=self.iterations,
          num_bands=num_bands,
          largest_group_size=group_size,
          sampling_prob=self.sampling_prob,
          truncated_batch_size=self.truncated_batch_size,
      )
    else:
      return accounting.amplified_bandmf_event(
          noise_multiplier=sigma,
          iterations=self.iterations,
          num_bands=num_bands,
          sampling_prob=self.sampling_prob,
      )

  def _check_calibrated(self) -> None:
    """Raises ValueError if noise_multiplier has not been set."""
    if self.noise_multiplier is None:
      raise ValueError(
          'noise_multiplier is not set. Call calibrate() or provide'
          ' noise_multiplier directly.'
      )

  def calibrate(
      self,
      *,
      epsilon: float,
      delta: float,
      tol: float | None = None,
      accountant_fn: AccountantFn = dp_accounting.pld.PLDAccountant,
  ) -> 'BandMFExecutionPlanConfig':
    """Returns a new config with a calibrated noise_multiplier.

    Args:
      epsilon: The target privacy budget.
      delta: The target privacy failure probability.
      tol: The tolerance in noise_multiplier space for the calibration binary
        search. Defaults to 1e-6 if not specified.
      accountant_fn: A function that returns a fresh privacy accountant used for
        calibration given a neighboring relation. Defaults to PLDAccountant.

    Returns:
      A new BandMFExecutionPlanConfig with calibrated noise_multiplier.
    """

    noise_multiplier = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: accountant_fn(self._neighboring_relation),
        make_event_from_param=self._get_dp_event,
        target_epsilon=epsilon,
        target_delta=delta,
        tol=tol,
    )

    return dataclasses.replace(  # pytype: disable=wrong-arg-types
        self, noise_multiplier=noise_multiplier
    )

  @classmethod
  def default(
      cls,
      num_bands: int,
      iterations: int,
      strategy_optimization_steps: int = 500,
      **kwargs,
  ) -> 'BandMFExecutionPlanConfig':
    """Returns a BandMFExecutionPlanConfig with an RMSE-optimized strategy.

    See BandMFExecutionPlanConfig for the full list of keyword arguments.

    Args:
      num_bands: The number of bands in the strategy matrix.
      iterations: The number of iterations the mechanism is defined for.
      strategy_optimization_steps: The number of optimization steps to use for
        the strategy matrix.
      **kwargs: Keyword arguments to pass to BandMFExecutionPlanConfig.

    Returns:
      A BandMFExecutionPlanConfig with an RMSE-optimized strategy.
    """
    strategy = toeplitz.optimize_banded_toeplitz(
        n=iterations,
        bands=num_bands,
        max_optimizer_steps=strategy_optimization_steps,
    )
    return BandMFExecutionPlanConfig(
        iterations=iterations, strategy=strategy, **kwargs
    )

  def make(
      self,
      performance_flags: PerformanceFlags | None = None,
  ) -> DPExecutionPlan:
    """Returns a DP execution plan for the given BandMF mechanism config.

    Args:
      performance_flags: Optional performance flags that control implementation
        details such as dtype, sharding, and microbatching. If None, default
        values are used for all performance flags.

    Returns:
      A DPExecutionPlan configured from this config and the given performance
      flags.

    Raises:
      ValueError: If noise_multiplier has not been set.
    """
    self._check_calibrated()
    if performance_flags is None:
      performance_flags = PerformanceFlags()

    @functools.wraps(clipping.clipped_grad)
    def clipped_grad_transform(*args, **kwargs):
      return clipping.clipped_grad(
          *args,
          **kwargs,
          l2_clip_norm=self.l2_clip_norm,
          normalize_by=self.normalize_by,
          rescale_to_unit_norm=self.rescale_to_unit_norm,
          dtype=performance_flags.dtype,
          microbatch_size=performance_flags.microbatch_size,
          spmd_axis_name=performance_flags.spmd_axis_name,
      )

    batch_selection_strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=self.sampling_prob,
        iterations=self.iterations,
        cycle_length=len(self.strategy),
        truncated_batch_size=self.truncated_batch_size,
        partition_type=self._partition_type,
    )

    max_column_norm = np.linalg.norm(self.strategy)
    column_normalize_for_n = self.iterations if self.column_normalize else None
    noising_matrix = toeplitz.inverse_as_streaming_matrix(
        self.strategy, column_normalize_for_n
    )

    query_sensitivity = clipped_grad_transform(lambda: None).sensitivity()

    dp_event = self._get_dp_event(self.noise_multiplier)

    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix,
        stddev=float(
            self.noise_multiplier * query_sensitivity * max_column_norm
        ),
        prng_key=performance_flags.noise_seed,
        dtype=performance_flags.dtype,
        intermediate_strategy=performance_flags.intermediate_strategy,
    )

    return DPExecutionPlan(
        clipped_grad=clipped_grad_transform,
        batch_selection_strategy=batch_selection_strategy,
        noise_addition_transform=privatizer,
        dp_event=dp_event,
        neighboring_relation=self._neighboring_relation,
    )
