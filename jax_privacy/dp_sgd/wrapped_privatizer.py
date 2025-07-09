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

"""A GradientComputer class that wraps a Privatizer."""

from collections.abc import Callable

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import typing
from jax_privacy.noise_addition import gradient_privatizer
import optax

logging.warning(
    'wrapped_privatizer.WrappedPrivatizer is deprecated and will be removed in '
    'Jax Privacy 1.5. Please use the Privatizer directly.'
)


class WrappedPrivatizer(
    # pytype: disable=not-indexable
    gradients.GradientComputer[
        typing.InputsT,
        typing.ParamsT,
        typing.ModelStateT,
        typing.NoiseStateT,
    ]
    # pytype: enable=not-indexable
):
  """Gradient computer that defers noising to a gradient privatizer.

  Note that the input Privatizer handles the rng state internally, and hence
  this `WrappedPrivatizer` ignores the rng passed to the instance method
  `add_noise_to_grads`.
  """

  def __init__(
      self,
      *,
      clipping_norm: float | None,
      rescale_to_unit_norm: bool,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      rng_per_param_fn: Callable[[chex.PRNGKey], chex.PRNGKey] = lambda x: x,
      global_norm_fn: Callable[[typing.ParamsT], jax.Array] = optax.global_norm,
      privatizer: gradient_privatizer.GradientPrivatizer,
  ):
    super().__init__(
        clipping_norm=clipping_norm,
        noise_multiplier=None,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=per_example_grad_method,
        rng_per_param_fn=rng_per_param_fn,
        global_norm_fn=global_norm_fn,
    )
    self._privatizer = privatizer

  def init_noise_state(self, params: typing.ParamsT) -> typing.NoiseStateT:
    """Initialize noise state for BandMF."""
    return self._privatizer.init(params)

  def add_noise_to_grads(
      self,
      grads: typing.ParamsT,
      rng_per_batch: chex.PRNGKey | None,
      total_batch_size: jax.Array,
      noise_state: typing.NoiseStateT,
  ) -> tuple[typing.ParamsT, jax.Array, typing.NoiseStateT]:
    if rng_per_batch is not None:
      logging.warning(
          'rng_per_batch is unused in WrappedPrivatizer. The rng is handled'
          ' internally by the privatizer. Set rng_per_batch to None to disable'
          ' this warning'
      )

    sum_of_clipped_grads = jax.tree.map(lambda x: x * total_batch_size, grads)

    noisy_grads, noise_state = self._privatizer.privatize(
        sum_of_clipped_grads=sum_of_clipped_grads,
        noise_state=noise_state,
    )

    noisy_grads = jax.tree.map(lambda x: x / total_batch_size, noisy_grads)
    noisy_grads: typing.ParamsT
    noise_state: typing.NoiseStateT
    # To comply with the GradientComputer API, we return the standard deviation
    # of the noise as a float32 scalar. Here we return NaN as a placeholder.
    return noisy_grads, jnp.nan, noise_state
