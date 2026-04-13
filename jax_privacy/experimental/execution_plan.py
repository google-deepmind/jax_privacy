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

**API Stability: 5/10 -- Subject to change!**

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

import copy
import dataclasses
import functools
from typing import Any, Callable

import dp_accounting
from jax_privacy import batch_selection
from jax_privacy import clipping
from jax_privacy import noise_addition
from jax_privacy.experimental import accounting
from jax_privacy.matrix_factorization import toeplitz
import numpy as np
import optax
import pydantic


NeighboringRelation = dp_accounting.NeighboringRelation


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
  """

  clipped_grad: Callable[..., clipping.BoundedSensitivityCallable]
  batch_selection_strategy: batch_selection.BatchSelectionStrategy
  noise_addition_transform: optax.GradientTransformation
  dp_event: dp_accounting.DpEvent


def _validate_epsilon_delta_noise_multiplier(
    epsilon: float | None,
    delta: float | None,
    noise_multiplier: float | None,
):
  """Validates exactly one of (epsilon, delta) or noise_multiplier is set."""
  if (epsilon is None) != (delta is None):
    raise ValueError('epsilon and delta must be either both None or both set.')
  if (epsilon is None) == (noise_multiplier is None):
    raise ValueError(
        'Exactly one of (epsilon, delta), and noise_multiplier must be set.'
    )


def _validate_adjacency_relation(
    accountant: dp_accounting.PrivacyAccountant,
    truncated_batch_size: int | None,
    partition_type: batch_selection.PartitionType,
) -> None:
  """Validates the adjacency relation is compatible with the config."""
  neighboring_relation = accountant.neighboring_relation

  if neighboring_relation is NeighboringRelation.ADD_OR_REMOVE_ONE:
    if partition_type != batch_selection.PartitionType.INDEPENDENT:
      raise ValueError(
          f'{neighboring_relation=} is only compatible with INDEPENDENT'
          f' partition_type, found {partition_type=}.'
      )
    if truncated_batch_size is not None:
      raise ValueError(
          f'{neighboring_relation=} does not support truncated_batch_size,'
          f' found {truncated_batch_size=}.'
      )

  else:
    if (
        partition_type == batch_selection.PartitionType.INDEPENDENT
        and truncated_batch_size is not None
    ):
      raise ValueError(
          f'{partition_type=} is not compatible with {truncated_batch_size=}'
      )


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(arbitrary_types_allowed=True),
)  # pytype: disable=wrong-keyword-args
class BandMFExecutionPlanConfig:
  """Configuration for an Amplified BandMF-based DPExecutionPlan.

  The expected batch size of the batch selection strategy is
  `num_examples / num_bands * sampling_prob`. In most cases, `num_examples` is
  ignored because `num_examples` is considered a sensitive quantity under some
  DP definitions. The exception is when using truncation, where it is necessary
  for accounting (hence, one should be careful about the DP definition when
  using truncation). The recommended way to configure this function is directly
  via (epsilon, delta), however for convenience it can also be configured via
  `noise_multiplier` by setting epsilon=delta=None.

  Example Usage (Standard DP-SGD):
  >>> config = BandMFExecutionPlanConfig.default(
  ...     num_bands=1,
  ...     iterations=1000,
  ...     epsilon=1.0,
  ...     delta=1e-5,
  ...     sampling_prob=0.1,
  ... )

  Example Usage (BandMF with default strategy matrix):
  >>> config = BandMFExecutionPlanConfig.default(
  ...     num_bands=4,
  ...     iterations=1000,
  ...     epsilon=1.0,
  ...     delta=1e-5,
  ...     sampling_prob=0.4,
  ... )

  Example Usage (BandMF with custom strategy):
  >>> config = BandMFExecutionPlanConfig(
  ...     strategy=np.array([1.0, 0.5, 0.2, 0.1]),
  ...     iterations=1000,
  ...     epsilon=1.0,
  ...     delta=1e-5,
  ...     sampling_prob=0.4,
  ... )

    - num_bands=1
    - sampling_prob = expected_batch_size / num_examples
    - partition_type = INDEPENDENT
    - neighboring_relation = ADD_OR_REMOVE_ONE
    - accountant = PLDAccountant(ADD_OR_REMOVE_ONE)

  This yields the usual Poisson sampling analysis, and uses uncorrelated
  Gaussian noise (the BandMF strategy degenerates to the identity matrix).

  References:
  - https://arxiv.org/abs/2306.08153
  - https://arxiv.org/abs/2405.15913

  Attributes:
    epsilon: The desired privacy budget.
    delta: Additional privacy parameter.
    noise_multiplier: If specified, gives the standard deviation of the
      uncorrelated gaussian noise used with the BandMF GradientPrivatizer.
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
      truncated_batch_size is set. Should not be provided if
      accountant.neighboring_relation is ADD_OR_REMOVE_ONE, since it is not
      public information.
    partition_type: How to partition the examples into groups for before Poisson
      sampling. EQUAL_SPLIT is the default, and is only compatible with zero-out
      and replace-one adjacency notions, while INDEPENDENT is compatible with
      the add-remove adjacency notion.
    dtype: The dtype to use for noise generation and gradient aggregation.
    column_normalize: Whether to column-normalize the strategy matrix.
    accountant: A privacy accountant that is used to calibrate the noise
      multiplier. Expected to have an empty state (or calibration may fail).
      Defaults to PLDAccountant with REPLACE_SPECIAL neighboring_relation.
    noise_seed: A seed for the random number generator used for noise addition.
  """

  iterations: int = pydantic.Field(ge=0)
  strategy: np.typing.ArrayLike = pydantic.Field()
  epsilon: float | None = pydantic.Field(ge=0, allow_inf_nan=True)
  delta: float | None = pydantic.Field(gt=0, le=1)
  noise_multiplier: float | None = pydantic.Field(default=None, ge=0)
  l2_clip_norm: float = pydantic.Field(default=1.0, ge=0)
  rescale_to_unit_norm: bool = pydantic.Field(default=True)
  normalize_by: float = pydantic.Field(default=1.0, ge=0)
  sampling_prob: float = pydantic.Field(default=1.0, ge=0, le=1)
  truncated_batch_size: int | None = pydantic.Field(default=None, ge=0)
  num_examples: int | None = pydantic.Field(default=None, ge=0)
  partition_type: batch_selection.PartitionType = pydantic.Field(
      default=batch_selection.PartitionType.EQUAL_SPLIT
  )
  dtype: Any = pydantic.Field(default=np.float32)
  column_normalize: bool = pydantic.Field(default=False)
  accountant: dp_accounting.PrivacyAccountant = pydantic.Field(
      default_factory=lambda: dp_accounting.pld.PLDAccountant(
          NeighboringRelation.REPLACE_SPECIAL
      )
  )
  noise_seed: int | None = None

  def __post_init__(self):
    _validate_epsilon_delta_noise_multiplier(
        self.epsilon, self.delta, self.noise_multiplier
    )
    if self.truncated_batch_size is not None and self.num_examples is None:
      raise ValueError('truncated_batch_size requires num_examples to be set.')
    _validate_adjacency_relation(
        self.accountant,
        self.truncated_batch_size,
        self.partition_type,
    )
    strategy = np.asarray(self.strategy)
    if strategy.ndim != 1:
      raise ValueError(f'strategy must be a 1D array, found {strategy.ndim}.')
    if strategy.size == 0 or strategy.size > self.iterations:
      raise ValueError(f'{strategy.size=} not in range [1, {self.iterations}].')

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

  @functools.cached_property
  def plan(self) -> DPExecutionPlan:
    """Returns a DP execution plan for the given BandMF mechanism config."""

    @functools.wraps(clipping.clipped_grad)
    def clipped_grad_transform(*args, **kwargs):
      return clipping.clipped_grad(
          *args,
          **kwargs,
          l2_clip_norm=self.l2_clip_norm,
          normalize_by=self.normalize_by,
          rescale_to_unit_norm=self.rescale_to_unit_norm,
          dtype=self.dtype,
      )

    query_sensitivity = clipped_grad_transform(lambda: None).sensitivity(
        self.accountant.neighboring_relation
    )

    noise_multiplier = self.noise_multiplier
    if noise_multiplier is None:
      make_fresh_accountant = functools.partial(copy.deepcopy, self.accountant)
      noise_multiplier = dp_accounting.calibrate_dp_mechanism(
          make_fresh_accountant=make_fresh_accountant,
          make_event_from_param=self._get_dp_event,
          target_epsilon=self.epsilon,
          target_delta=self.delta,
      )

    dp_event = self._get_dp_event(noise_multiplier)

    batch_selection_strategy = batch_selection.CyclicPoissonSampling(
        sampling_prob=self.sampling_prob,
        iterations=self.iterations,
        cycle_length=len(self.strategy),
        truncated_batch_size=self.truncated_batch_size,
        partition_type=self.partition_type,
    )

    max_column_norm = np.linalg.norm(self.strategy)
    column_normalize_for_n = self.iterations if self.column_normalize else None
    noising_matrix = toeplitz.inverse_as_streaming_matrix(
        self.strategy, column_normalize_for_n
    )

    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix,
        stddev=float(noise_multiplier * query_sensitivity * max_column_norm),
        prng_key=self.noise_seed,
        dtype=self.dtype,
    )

    return DPExecutionPlan(
        clipped_grad=clipped_grad_transform,
        batch_selection_strategy=batch_selection_strategy,
        noise_addition_transform=privatizer,
        dp_event=dp_event,
    )
