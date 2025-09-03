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

"""The updater computes and applies the update.

Typical usage:
  # Initialize model and optimizer (pmapped).
  params, network_state, opt_state, step_on_host = updater.init(
      rng=rng,
      inputs=inputs,
  )

  # Apply update (pmapped).
  params, network_state, opt_state, step_on_host, stats = updater.update(
      params=params,
      network_state=network_state,
      opt_state=opt_state,
      step_on_host=step_on_host,
      inputs=inputs,
  )
"""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Callable, Generic

import chex
import jax
from jax_privacy.dp_sgd import typing
import optax

StepOnHost = int
StepOnDevice = jax.Array


@chex.dataclass(frozen=True, kw_only=True)
class UpdaterState(
    Generic[typing.ParamsT, typing.ModelStateT, typing.NoiseStateT]
):
  """Container for the updater state."""

  params: typing.ParamsT
  network_state: typing.ModelStateT
  update_step: StepOnDevice
  opt_state: optax.OptState
  noise_state: typing.NoiseStateT = dataclasses.field(default_factory=dict)  # pylint: disable=invalid-field-call
  params_avg: Mapping[str, typing.ParamsT] = dataclasses.field(  # pylint: disable=invalid-field-call
      default_factory=dict)


class AbstractUpdater(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Generic[typing.InputsT, typing.ParamsT, typing.ModelStateT],
    metaclass=abc.ABCMeta,
    # pytype: enable=invalid-annotation
):
  """Defines and applies the update, potentially in parallel across devices."""

  @abc.abstractmethod
  def init(
      self,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[UpdaterState, StepOnHost]:
    """Provides initial training state.

    Args:
      rng: Random key.
      inputs: Training inputs.

    Returns:
      Initial updater state.
      Step maintained on the host to avoid device transfers.
    """
    raise NotImplementedError('init method is not implemented')

  @abc.abstractmethod
  def update(
      self,
      state: UpdaterState,
      inputs_producer: Callable[[], typing.InputsT],
      step_on_host: StepOnHost,
  ) -> tuple[UpdaterState, typing.Metrics, StepOnHost]:
    """Computes updated training state (to be pmapped).

    Args:
      state: Current updater state.
      inputs_producer: Producer of the training inputs.
      step_on_host: Step maintained on the host to avoid device transfers.

    Returns:
      state: New updater state.
      metrics: Metrics to log.
      step_on_host: Step incremented on host.
    """
    raise NotImplementedError('update method is not implemented')
