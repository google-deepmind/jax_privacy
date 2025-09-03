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

"""Defines train and evaluation functions that compute losses and metrics."""

import abc
from typing import Generic

import chex
from jax_privacy.dp_sgd import typing


class ForwardFn(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Generic[typing.InputsT, typing.ParamsT, typing.ModelStateT],
    # pytype: enable=invalid-annotation
    metaclass=abc.ABCMeta,
):
  """Defines forward passes for learning tasks."""

  @abc.abstractmethod
  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT]:
    """Initializes the model.

    Args:
      rng_key: random number generation key used for the random initialization.
      inputs: model inputs.
    Returns:
      Initialized model parameters and state.
    """

  @abc.abstractmethod
  def train_forward(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]]:
    """Forward pass per example (training time).

    Args:
      params: model parameters that should get updated during training.
      network_state: model state.
      rng_per_example: a random number generation key specific for a device and
        accumulation step. It can be used to create a unique seed per
        individual example by the user.
      inputs: model inputs.
    Returns:
      loss: loss function averaged on the mini-batch.
      aux:
        network_state: new model state
        metrics: metrics computed on the current mini-batch
    """

  @abc.abstractmethod
  def eval_forward(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.Metrics:
    """Forward pass per example (evaluation time).

    Args:
      params: model parameters that should get updated during training.
      network_state: model state.
      rng: random number generation key.
      inputs: model inputs.
    Returns:
      evaluation results for the mini-batch, as a pair of the form
      (per-example outputs, aggregated metrics)
    """
