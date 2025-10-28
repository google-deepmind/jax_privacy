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

"""Module for defining DP Execution Plans.

**API Stability: 5/10 -- Subject to change!**

This module introduces the `DPExecutionPlan`, an object designed to encapsulate
the core components of a differentially private (DP) mechanism. The primary aim
is to simplify the process of constructing and applying DP mechanisms by
packaging these components cohesively.

The design is framework-agnostic, specifying the essential pillars of a DP
mechanism—such as batch selection and noise addition—without tightly coupling
them to a specific training loop. Each component is exposed through a simple,
well-documented API, allowing for flexible integration into various frameworks
or direct use with JAX.

Constructors for these plans are highly configurable, offering access to the
full capabilities of the underlying components while also providing sensible
defaults. A key benefit is the assurance that, when used correctly, the
combination of these components will achieve the stated DP properties.

While the components are designed to work together, they can also be used
selectively. For instance, a researcher might choose to use only the noise
addition component if their dataset doesn't support the efficient random access
required by the batch selection strategy. Though this might invalidate the
formal DP guarantee, it can still be valuable for research or when a heuristic
quantification of privacy is acceptable. Ultimately, `DPExecutionPlan` aims to
free users to concentrate on their training pipeline setup, rather than on the
intricacies of correctly assembling DP components to achieve a desired privacy
guarantee.
"""

import abc
import copy
import dataclasses
import functools
import math

import dp_accounting
import jax.numpy as jnp
from jax_privacy.experimental import batch_selection
from jax_privacy.experimental import clipping
from jax_privacy.matrix_factorization import toeplitz
from jax_privacy.noise_addition import additive_privatizers
import optax
import pydantic

NeighboringRelation = dp_accounting.NeighboringRelation


@dataclasses.dataclass(frozen=True)
class DPExecutionPlan(abc.ABC):
  """Class for defining a DP execution plan.

  A DP execution plan consists of a collection of components which when used
  together in the expected manner determine the DP guarantee, along with a
  DpEvent which precisely quantifies it. If constructed via one of the
  ExecutionPlanConfig classes defined in this module, then the `dp_event` can
  be trusted as having been formally verified by the JAX Privacy authors.

  In pseudo-code, the components of this dataclass should roughly be used as
  follows:

  .. code-block:: python

    noise_state = noise_addition_transform.init(...)
    for indices in batch_selection_strategy():
      batch = data.select(indices)
      clipped_grad = clipped_aggregation_fn(batch, ...)
      dp_grad, noise_state = noise_addition_transform.update(
          clipped_grad, noise_state
      )
      # Sensitive, discard immediately after use.
      del indices, batch, clipped_grad
      # Arbitrary post-processing of dp_grad.

  If possible, we recommend coupling the batch selection, clipped aggregation,
  and noise addition components as tightly as possible to ensure sensitive
  objects are not intercepted and used unintentionally.

  Attributes:
    clipped_aggregation_fn: A bounded sensitivity function, such as one that
      computes a sum of per-example clipped gradients.
    batch_selection_strategy: Determines how batches are formed in each
      iteration.
    noise_addition_transform: Stateful transformation that adds noise to clipped
      and aggregated gradients after each iteration.
    dp_event: Characterizes the mechanism in terms of primitive building blocks
      that dp_accounting knows how to analyze.
  """

  @abc.abstractmethod
  def clipped_grad(
      self, fun, argnums, has_aux, **kwargs
  ) -> clipping.BoundedSensitivityCallable:
    """Creates a function that evaluates the sum-of-clipped gradients of fun."""

  @abc.abstractmethod
  def batch_selection_strategy(
      self, **kwargs
  ) -> batch_selection.BatchSelectionStrategy:
    """Returns the batch selection strategy."""

  @abc.abstractmethod
  def noise_addition_transform(self, **kwargs) -> optax.GradientTransformation:
    """Returns the noise addition transform."""

  @property
  @abc.abstractmethod
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DpEvent for the execution plan."""


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


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(arbitrary_types_allowed=True),
)  # pytype: disable=wrong-keyword-args
class BandMFExecutionPlan(DPExecutionPlan):
  """Configuration for a BandMF-based DPExecutionPlan.

  The expected batch size of the batch selection strategy is
  `num_examples / num_bands * sampling_prob`. In most cases, `num_examples' is
  ignored because `num_examples` is considered a sensitive quantity under some
  DP definitions. The exception is when using truncation, where it is necessary
  for accounting (hence, one should be careful about the DP definition when
  using truncation). The recommended way to configure this function is directly
  via (epsilon, delta), however for convenience it can also be configured via
  `noise_multiplier` by setting epsilon=delta=None.

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
    num_bands: The number of bands in the BandMF strategy matrix.
    l2_clip_norm: The maximum L2 norm of the per-example gradients.
    rescale_to_unit_norm: Divide the clipped gradient by the l2_clip_norm.
    normalize_by: Divide the sum-of-clipped gradients by this value.
    sampling_prob: The Poisson sampling probability for each example in a group.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to.
    num_examples: The number of examples in the dataset. Unused if
      truncated_batch_size is None.
    use_fixed_size_groups: Whether to discard examples so that all groups have
      the same size before sampling. If sampling_prob=1, this guarantees that
      the batch selection strategy will produce fixed-size batches.
    accountant: A privacy accountant that is used to calibrate the noise
      multiplier. Expected to have an empty state (or calibration may fail).
      Defaults to PLDAccountant with REPLACE_SPECIAL neighboring_relation.
    neighboring_relation: The neighboring relation to use for the accountant.
      Defaults to REPLACE_SPECIAL. Must be consistent with the accountant and
      the arguments passed to the accountant.
  """

  # privacy parameters
  epsilon: float | None = pydantic.Field(ge=0, allow_inf_nan=True)
  delta: float | None = pydantic.Field(gt=0, le=1)
  noise_multiplier: float | None = pydantic.Field(default=None, ge=0)

  # noise addition parameters
  iterations: int = pydantic.Field(ge=0)
  num_bands: int = pydantic.Field(ge=1)

  # gradient clipping parameters
  l2_clip_norm: float = pydantic.Field(ge=0, default=1.0)
  rescale_to_unit_norm: bool = True
  normalize_by: float = pydantic.Field(gt=0, default=1.0)

  # batch selection parameters
  sampling_prob: float = pydantic.Field(default=1.0, ge=0, le=1)
  truncated_batch_size: int | None = pydantic.Field(default=None, ge=0)
  num_examples: int | None = pydantic.Field(default=None, ge=0)
  use_fixed_size_groups: bool = False

  # accountant parameters
  accountant: dp_accounting.PrivacyAccountant = pydantic.Field(
      default_factory=lambda: dp_accounting.pld.PLDAccountant(
          NeighboringRelation.REPLACE_SPECIAL
      )
  )
  neighboring_relation: dp_accounting.NeighboringRelation = (
      NeighboringRelation.REPLACE_SPECIAL
  )

  def __post_init__(self):
    _validate_epsilon_delta_noise_multiplier(
        self.epsilon, self.delta, self.noise_multiplier
    )
    if self.accountant.neighboring_relation != self.neighboring_relation:
      raise ValueError(
          'neighboring_relation must match the accountant. Found '
          f'{self.accountant.neighboring_relation, self.neighboring_relation}.'
      )
    if (
        self.truncated_batch_size is not None
        and self.neighboring_relation is NeighboringRelation.ADD_OR_REMOVE_ONE
    ):
      raise ValueError(
          'truncated_batch_size with ADD_OR_REMOVE_ONE is not supported.'
      )
    if (
        self.num_bands != 1
        and self.neighboring_relation is NeighboringRelation.ADD_OR_REMOVE_ONE
    ):
      # TODO: b/415360727 - This can be fixed by using different partitionings.
      raise ValueError(f'{self.neighboring_relation=} requires num_bands=1.')

  def _get_dp_event(self, sigma: float) -> dp_accounting.DpEvent:
    """Returns a DpEvent for the BandMF mechanism."""
    # Theorem 5 of https://arxiv.org/pdf/2306.08153. See also Theorem 1.
    if self.truncated_batch_size:
      # Larger groups have higher probability of truncation (worse privacy).
      largest_group_size = (
          self.num_examples // self.num_bands
          if self.use_fixed_size_groups
          else math.ceil(self.num_examples / self.num_bands)
      )
      single_cycle_event = dp_accounting.TruncatedSubsampledGaussianDpEvent(
          dataset_size=largest_group_size,
          sampling_probability=self.sampling_prob,
          truncated_batch_size=self.truncated_batch_size,
          noise_multiplier=sigma,
      )
    else:
      single_cycle_event = dp_accounting.PoissonSampledDpEvent(
          self.sampling_prob,
          dp_accounting.GaussianDpEvent(noise_multiplier=sigma),
      )
    return dp_accounting.SelfComposedDpEvent(
        single_cycle_event, math.ceil(self.iterations / self.num_bands)
    )

  @functools.cached_property
  def calibrated_noise_multiplier(self) -> float:
    """Returns the noise multiplier; if not set, calibrates it first."""
    if self.noise_multiplier is not None:
      return self.noise_multiplier
    return dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=functools.partial(copy.deepcopy, self.accountant),
        make_event_from_param=self._get_dp_event,
        target_epsilon=self.epsilon,
        target_delta=self.delta,
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return self._get_dp_event(self.calibrated_noise_multiplier)

  def clipped_grad(
      self, fun, argnums=0, has_aux=False, **kwargs
  ) -> clipping.BoundedSensitivityCallable:
    return clipping.clipped_grad(
        fun,
        argnums=argnums,
        has_aux=has_aux,
        l2_clip_norm=self.l2_clip_norm,
        rescale_to_unit_norm=self.rescale_to_unit_norm,
        normalize_by=self.normalize_by,
        **kwargs,
    )

  def noise_addition_transform(
      self, strategy_opt_kwargs=None, privatizer_kwargs=None
  ) -> optax.GradientTransformation:
    strategy_opt_kwargs = strategy_opt_kwargs or {}
    privatizer_kwargs = privatizer_kwargs or {}
    query = self.clipped_grad(lambda: None)

    # dp_accounting.GaussianDpEvent defines noise_multiplier w.r.t.
    # l2_norm_bound rather than sensitivity.
    norm_bound = query.l2_norm_bound

    # 1D vector of Toeplitz coefficients.
    mf_strategy = toeplitz.optimize_banded_toeplitz(
        n=self.iterations, bands=self.num_bands, **strategy_opt_kwargs
    )
    max_col_norm = jnp.linalg.norm(mf_strategy)
    stdev = float(self.calibrated_noise_multiplier * norm_bound * max_col_norm)
    noising_matrix = toeplitz.inverse_as_streaming_matrix(mf_strategy)

    return additive_privatizers.matrix_factorization_privatizer(
        noising_matrix, stddev=stdev, **privatizer_kwargs
    )

  def batch_selection_strategy(
      self, **kwargs
  ) -> batch_selection.BatchSelectionStrategy:
    return batch_selection.CyclicPoissonSampling(
        sampling_prob=self.sampling_prob,
        iterations=self.iterations,
        cycle_length=self.num_bands,
        truncated_batch_size=self.truncated_batch_size,
        **kwargs,
    )
