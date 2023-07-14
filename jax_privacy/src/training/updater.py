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

"""The updater computes and applies the update.

Typical usage:
  # Initialize model and optimizer (pmapped).
  params, network_state, opt_state, step_count = updater.init(
      rng=rng,
      inputs=inputs,
  )

  # Apply update (pmapped).
  params, network_state, opt_state, step_count, stats = updater.update(
      params=params,
      network_state=network_state,
      opt_state=opt_state,
      step_count=step_count,
      inputs=inputs,
  )
"""

import abc
from typing import Generic, NamedTuple

import chex
from jax_privacy.src.dp_sgd import typing
import optax


class StepCount(NamedTuple):
  """Hierarchical step - a full batch count plus inner accumulation step."""

  update_step: int
  accumulation_step: int

  def next(self, every_k: int) -> 'StepCount':
    """Returns updated with by accumulation step, rolling over every k."""
    new_accumulation_step = self.accumulation_step + 1
    return StepCount(
        update_step=(self.update_step + new_accumulation_step // every_k),
        accumulation_step=(new_accumulation_step % every_k),
    )


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
  ) -> tuple[
      typing.ParamsT, typing.ModelStateT, optax.MultiStepsState, StepCount]:
    """Provides initial training state.

    Args:
      rng: Random key.
      inputs: Training inputs.

    Returns:
      params: Initial model parameters (both trainable and frozen).
      network_state: Initial network state.
      opt_state: Initial optimiser state.
      step_count: Initial number of full steps and inner accumulation steps.
    """
    raise NotImplementedError('init method is not implemented')

  @abc.abstractmethod
  def update(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      opt_state: optax.MultiStepsState,
      step_count: StepCount,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT, optax.MultiStepsState,
             StepCount, typing.Metrics]:
    """Computes updated training state (to be pmapped).

    Args:
      params: Model parameters (both trainable and frozen).
      network_state: Network state.
      opt_state: Optimiser state.
      step_count: Number of full steps and inner accumulation steps.
      inputs: Training inputs.

    Returns:
      params: Updated model parameters (both trainable and frozen).
      network_state: Updated network state.
      opt_state: Updated optimiser state.
      step_count: Updated number of full steps and inner accumulation steps.
      scalars: Scalar outputs to log.
    """
    raise NotImplementedError('update method is not implemented')

  @abc.abstractmethod
  def step_count_from_opt_state(
      self,
      opt_state: optax.MultiStepsState,
  ) -> StepCount:
    """Returns the hierarchical step number."""

  @abc.abstractmethod
  def evaluate(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.Metrics:
    """Evaluates the model with the current state.

    Args:
      params: Model parameters (both trainable and frozen).
      network_state: Network state.
      rng: Random key.
      inputs: Evaluation inputs, consisting of tensors of shape
        (num_local_replicas, batch_size, ...).

    Returns:
      Evaluation results for the mini-batch, as a pair of the form
      (per-example outputs over all hosts, aggregated metrics).
      The per-example outputs have shape (num_replicas, batch_size, ...).
    """

  @abc.abstractmethod
  def optimizer(self) -> optax.GradientTransformation:
    """Returns optimiser giving rise to `opt_state`."""

  @abc.abstractmethod
  def init_average(
      self,
      params: typing.ParamsT,
  ) -> typing.ParamsT:
    """Initialises a copy of the params for moving averages.

    Taking a copy is important because `params` may subsequently be donated.

    Args:
      params: Model parameters (both trainable and frozen).

    Returns:
      Initial averages of model parameters (both trainable and frozen).
    """

  @abc.abstractmethod
  def update_ema(
      self,
      ema_params: typing.ParamsT,
      params: typing.ParamsT,
      opt_state: optax.MultiStepsState,
      *,
      mu: chex.Numeric,
      start_step: chex.Numeric,
  ) -> typing.ParamsT:
    """Initialises a copy of the params for exponential moving averages.

    Taking a copy is important because `params` may subsequently be donated.

    Args:
      ema_params: Existing averages of parameters (both trainable and frozen).
      params: Model parameters (both trainable and frozen).
      opt_state: Optimiser state.
      mu: Decay factor.
      start_step: Update step number at which to start applying averaging.

    Returns:
      Updated averages of model parameters (both trainable and frozen).
    """

  @abc.abstractmethod
  def update_polyak(
      self,
      polyak_params: typing.ParamsT,
      params: typing.ParamsT,
      opt_state: optax.MultiStepsState,
      *,
      start_step: chex.Numeric,
  ) -> typing.ParamsT:
    """Initialises a copy of the params for Polyak moving averages.

    Taking a copy is important because `params` may subsequently be donated.

    Args:
      polyak_params: Existing averages of parameters (both trainable and
        frozen).
      params: Model parameters (both trainable and frozen).
      opt_state: Optimiser state.
      start_step: Update step number at which to start applying averaging.

    Returns:
      Updated averages of model parameters (both trainable and frozen).
    """
