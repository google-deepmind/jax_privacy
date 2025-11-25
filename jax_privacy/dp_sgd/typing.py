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

"""General types re-used across the codebase.

Sub-directories may tighten the typing by using more restrictive types than the
general Inputs, ModelState and Params defined here.
"""

# pylint: disable=invalid-field-call  # fails for chex.dataclass

from collections.abc import Mapping
import dataclasses
from typing import Callable, Generic, Literal, Protocol, TypeVar

import chex
import jax
from jaxtyping import Array, Float  # pylint: disable=g-multiple-import, g-importing-member


GradNorm = jax.Array
GradNormPerExample = jax.Array
Loss = jax.Array
NumpyMetrics = dict[str, chex.ArrayNumpy]  # matches JAXline expectation

InputsT = TypeVar('InputsT', bound=chex.ArrayTree)
ModelStateT = TypeVar('ModelStateT', bound=chex.ArrayTree)
ParamsT = TypeVar('ParamsT', bound=chex.ArrayTree)
NoiseStateT = TypeVar('NoiseStateT', bound=chex.ArrayTree)

SquareMatrix = Float[Array, 'n n']

AutoTuneField = Literal[
    'batch_size',
    'noise_multiplier',
    'num_updates',
    'epsilon',
    None,
]


@chex.dataclass
class Metrics:
  """Container for various metrics."""

  scalars_avg: Mapping[str, chex.Numeric] = dataclasses.field(
      default_factory=dict
  )
  scalars_sum: Mapping[str, chex.Numeric] = dataclasses.field(
      default_factory=dict
  )
  scalars_last: Mapping[str, chex.Numeric] = dataclasses.field(
      default_factory=dict
  )
  per_example: Mapping[str, chex.Numeric] = dataclasses.field(
      default_factory=dict
  )

  @property
  def scalars(self) -> Mapping[str, chex.Numeric]:
    return {**self.scalars_avg, **self.scalars_sum, **self.scalars_last}


NormFn = Callable[[ParamsT], jax.Array]
GradClippingFn = Callable[[ParamsT], tuple[ParamsT, jax.Array]]


class LossFn(Protocol, Generic[InputsT, ParamsT, ModelStateT]):

  def __call__(
      self,
      params: ParamsT,
      network_state: ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: InputsT,
  ) -> tuple[Loss, tuple[ModelStateT, Metrics]]:
    """Computes the loss function.

    Args:
      params: Trainable parameters.
      network_state: Network state.
      rng_per_example: a random number generation key specific for a device and
        accumulation step. It can be used to create a unique seed per individual
        example by the user.
      inputs: Model inputs.

    Returns:
      Tuple consisting of (loss, aux).
    """


class ValueAndGradFn(Protocol, Generic[InputsT, ParamsT, ModelStateT]):

  def __call__(
      self,
      params: ParamsT,
      network_state: ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: InputsT,
  ) -> tuple[tuple[Loss, tuple[ModelStateT, Metrics]], ParamsT]:
    """Computes (potentially clipped) gradients.

    Args:
      params: Trainable parameters.
      network_state: Network state.
      rng_per_example: a random number generation key specific for a device and
        accumulation step. It can be used to create a unique seed per individual
        example by the user.
      inputs: Model inputs.

    Returns:
      Value, auxiliary outputs, and (potentially clipped) gradients.
    """
