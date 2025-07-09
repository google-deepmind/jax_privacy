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

"""The evaluator computes evaluation metrics.

Typical usage:
  # The evaluator requires a forward function and an RNG:
  evaluator = evaluator.Evaluator(
        forward_fn=forward_fn,  # see `forward.py`
        rng=jax.random.PRNGkey(42),
  )

  ...

  # Evaluate whole dataset.
    metrics = evaluator.evaluate_dataset(
        updater_state=updater_state, ds_iterator=ds_iterator
    )
"""

import abc
from collections.abc import Iterator, Mapping
from typing import Any, Generic, Protocol

import chex
import haiku as hk
import jax
from jax_privacy.dp_sgd import typing
from jax_privacy.training import devices
from jax_privacy.training import forward
from jax_privacy.training import updater
from jaxline import utils as jaxline_utils


def _to_scalar(
    x: chex.Numeric | chex.ArrayNumpy,
) -> chex.Numeric | chex.ArrayNumpy:
  """Casts the input to a scalar if it is an array with a single element."""
  if isinstance(x, (chex.Array, chex.ArrayNumpy)) and x.size == 1:
    return x.reshape(())
  else:
    return x


class Evaluator(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Protocol[typing.InputsT],
    # pytype: enable=invalid-annotation
):
  """Defines the evaluation, potentially in parallel across devices."""

  def evaluate_dataset(
      self,
      updater_state: updater.UpdaterState,
      ds_iterator: Iterator[typing.InputsT],
  ) -> Mapping[str, Any]:
    """Evaluates a dataset with the given state."""


class AbstractEvaluator(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Generic[typing.InputsT],
    # pytype: enable=invalid-annotation
    metaclass=abc.ABCMeta,
):
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      forward_fn: forward.ForwardFn,
      rng: chex.PRNGKey,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
  ):

    self._forward_fn = forward_fn
    self._rng = rng
    self._device_layout = device_layout

    self._pmapped_evaluate = jax.pmap(
        self._single_device_evaluate,
        in_axes=(0, 0, 0, 0),  # params, net_state, inputs, step (on device)
        donate_argnums=(2,),
        **device_layout.pmap_kwargs)

  def _evaluate_batch(
      self,
      updater_state: updater.UpdaterState,
      inputs: typing.InputsT,
  ) -> Mapping[str, typing.Metrics]:
    """Evaluates model parameters."""
    # The function below is p-mapped, so arguments must be provided without name
    # and in the right order.
    def eval_params(params: typing.ParamsT) -> typing.Metrics:
      metrics = self._pmapped_evaluate(
          params,
          updater_state.network_state,
          inputs,
          updater_state.update_step,
      )
      return jaxline_utils.get_first(metrics)

    results = {}
    results['last'] = eval_params(updater_state.params)
    for name, params_avg in updater_state.params_avg.items():
      params = hk.data_structures.merge(updater_state.params, params_avg)
      results[name] = eval_params(params)
    return results

  def _single_device_evaluate(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      inputs: typing.InputsT,
      update_step: updater.StepOnDevice,
  ) -> typing.Metrics:
    """Evaluates model parameters (to be pmapped)."""
    rng = jax.random.fold_in(self._rng, update_step)
    # Note on rngs:
    # - rng is common across replicas.
    # - rng_per_example is specialised per sample (for independent randonmness).
    rng_per_example = jax.random.fold_in(rng, self._device_layout.replica_index)

    metrics = self._forward_fn.eval_forward(
        params, network_state, rng_per_example, inputs)
    per_example = jax.lax.all_gather(
        metrics.per_example, **self._device_layout.data_psum_kwargs)
    scalars_avg = jax.lax.pmean(
        metrics.scalars_avg, **self._device_layout.data_psum_kwargs)
    scalars_sum = jax.lax.psum(
        metrics.scalars_sum, **self._device_layout.data_psum_kwargs)
    return typing.Metrics(
        scalars_avg=scalars_avg,
        scalars_sum=scalars_sum,
        per_example=per_example,
    )

  @abc.abstractmethod
  def evaluate_dataset(
      self,
      updater_state: updater.UpdaterState,
      ds_iterator: Iterator[typing.InputsT],
  ) -> Mapping[str, Any]:
    """Evaluates a dataset with the given state."""
