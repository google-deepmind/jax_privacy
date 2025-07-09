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

import copy
import dataclasses
import functools
import math

import dp_accounting
import jax.numpy as jnp
from jax_privacy.experimental import batch_selection
from jax_privacy.experimental import clipping
from jax_privacy.matrix_factorization import toeplitz
from jax_privacy.noise_addition import distributed_noise_generation as dng
from jax_privacy.noise_addition import gradient_privatizer
import pydantic


_REPLACE_SPECIAL = dp_accounting.NeighboringRelation.REPLACE_SPECIAL


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

  ```python
  noise_state = noise_addition_transform.init(...)
  for indices in batch_selection_strategy():
    batch = data.select(indices)
    clipped_grad = clipped_aggregation_fn(batch, ...)
    dp_grad, noise_state = noise_addition_transform.update(
        clipped_grad, noise_state
    )
    del indices, batch, clipped_grad # Sensitive, discard immediately after use.
    # Arbitrary post-processing of dp_grad.
  ```

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

  clipped_aggregation_fn: clipping.BoundedSensitivityCallable
  batch_selection_strategy: batch_selection.BatchSelectionStrategy
  noise_addition_transform: gradient_privatizer.GradientPrivatizer
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


@pydantic.dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    config=pydantic.ConfigDict(arbitrary_types_allowed=True),
)  # pytype: disable=wrong-keyword-args
class BandMFExecutionPlanConfig:
  """Configuration for a BandMF-based DPExecutionPlan.

  The expected batch size of the batch selection strategy is
  `num_examples / num_bands * sampling_prob`. This is not an input to the
  function because `num_examples` is considered a sensitive quantity under some
  DP definitions. The recommended way to configure this function is directly
  via (epsilon, delta), however for convenience it can also be configured via
  `noise_multiplier` by setting epsilon=delta=None.

  References:
  - https://arxiv.org/abs/2306.08153
  - https://arxiv.org/abs/2405.15913

  Attributes:
    num_examples: The number of examples in the dataset.
    iterations: The number of iterations the mechanism is defined for. Tip: Set
      this to be a multiple of num_bands for the best utility.
    num_bands: The number of bands in the BandMF strategy matrix.
    epsilon: The desired privacy budget.
    delta: Additional privacy parameter.
    noise_multiplier: If specified, gives the standard devaiation of the
      uncorrelated gaussian noise used with the BandMF GradientPrivatizer.
    sampling_prob: The Poisson sampling probability for each example in a group.
    shuffle: Whether to shuffle the data before partitioning it.
    use_fixed_size_groups: Whether to discard examples so that all groups have
      the same size before sampling. If sampling_prob=1, this guarantees that
      the batch selection strategy will produce fixed-size batches.
    strategy_optimization_steps: Number of strategy optimization steps.
    accountant: A privacy accountant that is used to calibrate the noise
      multiplier. Expected to have an empty state (or calibration may fail).
      Defaults to PLDAccountant with REPLACE_SPECIAL neighboring_relation.
  """

  num_examples: int = pydantic.Field(ge=0)
  iterations: int = pydantic.Field(ge=0)
  num_bands: int = pydantic.Field(ge=1)
  epsilon: float | None = pydantic.Field(ge=0, allow_inf_nan=True)
  delta: float | None = pydantic.Field(gt=0, le=1)
  noise_multiplier: float | None = pydantic.Field(ge=0)
  sampling_prob: float = pydantic.Field(default=1.0, ge=0, le=1)
  shuffle: bool = False
  use_fixed_size_groups: bool = False
  strategy_optimization_steps: int = 500
  accountant: dp_accounting.PrivacyAccountant = pydantic.Field(
      default_factory=lambda: dp_accounting.pld.PLDAccountant(_REPLACE_SPECIAL)
  )

  def __post_init__(self):
    _validate_epsilon_delta_noise_multiplier(
        self.epsilon, self.delta, self.noise_multiplier
    )

  def _get_dp_event(self, sigma: float) -> dp_accounting.DpEvent:
    # Theorem 5 of https://arxiv.org/pdf/2306.08153. See also Theorem 1.
    single_cycle_event = dp_accounting.PoissonSampledDpEvent(
        self.sampling_prob,
        dp_accounting.GaussianDpEvent(noise_multiplier=sigma),
    )
    return dp_accounting.SelfComposedDpEvent(
        single_cycle_event, math.ceil(self.iterations / self.num_bands)
    )

  def make(
      self,
      clipped_aggregation_fn: clipping.BoundedSensitivityCallable,
  ) -> DPExecutionPlan:
    """Returns a DP execution plan for the given BandMF mechanism config."""

    query_sensitivity = clipped_aggregation_fn.sensitivity(
        self.accountant.neighboring_relation
    )
    noise_multiplier = self.noise_multiplier
    if noise_multiplier is None:
      noise_multiplier = dp_accounting.calibrate_dp_mechanism(
          make_fresh_accountant=functools.partial(copy.copy, self.accountant),
          make_event_from_param=self._get_dp_event,
          target_epsilon=self.epsilon,
          target_delta=self.delta,
      )

    dp_event = self._get_dp_event(noise_multiplier)

    batch_selection_strategy = batch_selection.CyclicPoissonSampling(
        self.num_examples,
        sampling_prob=self.sampling_prob,
        iterations=self.iterations,
        cycle_length=self.num_bands,
        shuffle=self.shuffle,
        even_partition=self.use_fixed_size_groups,
    )

    # 1D vector of Toeplitz coefficients.
    mf_strategy = toeplitz.optimize_banded_toeplitz(
        n=self.iterations,
        bands=self.num_bands,
        max_optimizer_steps=self.strategy_optimization_steps,
    )
    max_column_norm = jnp.linalg.norm(mf_strategy)
    noising_matrix = toeplitz.inverse_as_streaming_matrix(mf_strategy)

    noise_addition_transform = (
        dng.streaming_matrix_to_single_machine_privatizer(
            noising_matrix,
            stddev=float(
                noise_multiplier * query_sensitivity * max_column_norm
            ),
        )
    )

    return DPExecutionPlan(
        clipped_aggregation_fn=clipped_aggregation_fn,
        batch_selection_strategy=batch_selection_strategy,
        noise_addition_transform=noise_addition_transform,
        dp_event=dp_event,
    )
