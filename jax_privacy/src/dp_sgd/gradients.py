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

"""Differentially private gradient computation."""

from typing import Callable, Generic, Optional

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import devices
from jax_privacy.src.dp_sgd import grad_clipping
from jax_privacy.src.dp_sgd import optim
from jax_privacy.src.dp_sgd import typing
import optax


class GradientComputer(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Generic[typing.InputsT, typing.ParamsT, typing.ModelStateT]
    # pytype: enable=invalid-annotation
):
  """Computes (potentially) clipped and noisy gradients."""

  def __init__(
      self,
      *,
      clipping_norm: Optional[float],
      noise_multiplier: Optional[float],
      rescale_to_unit_norm: bool,
      vectorize_grad_clipping: bool,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
      rng_per_param_fn: Callable[[chex.PRNGKey], chex.PRNGKey] = lambda x: x,
      global_norm_fn: typing.NormFn = optax.global_norm,
  ):
    """Initialises the gradient computation.

    Args:
      clipping_norm: maximum L2 norm to which the input tree should be clipped.
      noise_multiplier: standard deviation of the noise to add to the average
         of the clipped gradient to make it differentially private. It will be
         multiplied by `clipping_norm / total_batch_size` before the noise gets
         actually added.
      rescale_to_unit_norm: whether the tree should be rescaled to have an L2
        norm of one once it got clipped.
      vectorize_grad_clipping: Whether to use the `vmap` version of gradient
        clipping (as opposed to an unrolled loop). This is faster, but uses
        more memory.
      device_layout: Common args to `pmap` and `psum` for data parallelism.
      rng_per_param_fn: Optional callable to allow gradient noise random keys
        to be specialised for different param slices.
      global_norm_fn: function to compute the L2 norm of an ArrayTree.
    """
    self._clipping_norm = clipping_norm
    self._noise_multiplier = noise_multiplier
    self._rescale_to_unit_norm = rescale_to_unit_norm
    self._vectorize_grad_clipping = vectorize_grad_clipping
    self._device_layout = device_layout
    self._rng_per_param_fn = rng_per_param_fn
    self._global_norm_fn = global_norm_fn

  @property
  def clipping_norm(self):
    return self._clipping_norm

  @property
  def using_clipped_grads(self):
    return self.clipping_norm not in (float('inf'), None)

  def global_norm(self, x: chex.ArrayTree) -> chex.ArrayTree:
    return self._global_norm_fn(x)

  def clean_gradients(
      self,
      loss_fn: typing.LossFn,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_batch: chex.PRNGKey,
      accumulation_step: chex.Array,
      inputs: typing.InputsT,
  ) -> typing.ParamsT:
    """Computes unclipped gradients of the given loss function.

    Args:
      loss_fn: Loss function whose gradients are required.
      params: Trainable parameters.
      network_state: Network state input to `loss_fn`.
      rng_per_batch: Random number key, expected to be common across devices
        and across micro-batches constituting the same logical batch.
      accumulation_step: Micro-batch number within a logical batch.
      inputs: Inputs to `loss_fn`.

    Returns:
      Unclipped gradients.
    """
    rng_per_example = self._rng_per_example(rng_per_batch, accumulation_step)

    # Compute gradients of the loss function w.r.t. the parameters.
    device_grads, unused_aux = jax.grad(loss_fn, has_aux=True)(
        params, network_state, rng_per_example, inputs)
    avg_grads = jax.lax.pmean(
        device_grads, **self._device_layout.data_psum_kwargs)

    return avg_grads

  def loss_and_clipped_gradients(
      self,
      loss_fn: typing.LossFn,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_batch: chex.PRNGKey,
      accumulation_step: chex.Array,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    """Computes (potentially) clipped gradients of the given loss function.

    Args:
      loss_fn: Loss function whose clipped gradients are required.
      params: Trainable parameters.
      network_state: Network state input to `loss_fn`.
      rng_per_batch: Random number key, expected to be common across devices
        and across micro-batches constituting the same logical batch.
      accumulation_step: Micro-batch number within a logical batch.
      inputs: Inputs to `loss_fn`.

    Returns:
      Tuple consisting of (loss-and-aux, clipped_grads)
      where `loss-and-aux` is as is returned by `loss_fn` (with the addition
      of the grad norm per example in the metrics).
    """
    rng_per_example = self._rng_per_example(rng_per_batch, accumulation_step)

    # Compute clipped-per-example gradients of the loss function w.r.t. the
    # parameters.
    (loss, (network_state, metrics)), device_grads = (
        self.value_and_clipped_grad(jax.value_and_grad(loss_fn, has_aux=True))(
            params, network_state, rng_per_example, inputs))

    # Synchronize metrics and gradients across devices.
    loss, metrics_avg, avg_grads = jax.lax.pmean(
        (loss, metrics.scalars_avg, device_grads),
        **self._device_layout.data_psum_kwargs,
    )
    metrics_sum = jax.lax.psum(
        metrics.scalars_sum,
        **self._device_layout.data_psum_kwargs,
    )
    metrics_per_example = jax.lax.all_gather(
        metrics.per_example,
        **self._device_layout.data_psum_kwargs,
        tiled=True,
    )

    metrics = typing.Metrics(
        scalars_avg=metrics_avg,
        scalars_sum=metrics_sum,
        per_example=metrics_per_example,
    )
    return (loss, (network_state, metrics)), avg_grads

  def value_and_clipped_grad(
      self,
      value_and_grad_fn: typing.ValueAndGradFn,
  ) -> typing.ValueAndGradFn:
    """Creates the function commputing (potentially) clipped gradients.

    Args:
      value_and_grad_fn: Function that produces unclipped gradients.
        It is expected to have the following signature:
        `(loss, aux), grad = grad_fn(params, inputs, network_state, rng_key)`.

    Returns:
      A function computing gradients that are potentially clipped per sample.
    """
    if not self.using_clipped_grads:
      if self._rescale_to_unit_norm:
        raise ValueError('Cannot rescale to unit norm without clipping.')
      return value_and_grad_fn

    clipping_fn = grad_clipping.global_clipping(
        global_norm_fn=self._global_norm_fn,
        clipping_norm=self._clipping_norm,
        rescale_to_unit_norm=self._rescale_to_unit_norm,
    )

    if self._vectorize_grad_clipping:
      # Compute gradients clipped per sample using vectorization.
      return grad_clipping.value_and_clipped_grad_vectorized(
          value_and_grad_fn,
          clipping_fn=clipping_fn)
    else:
      # Compute gradients clipped per sample using a (JAX) loop.
      return grad_clipping.value_and_clipped_grad_loop(
          value_and_grad_fn,
          clipping_fn=clipping_fn)

  def _rng_per_example(
      self,
      rng_per_batch: chex.PRNGKey,
      accumulation_step: chex.Array,
  ) -> chex.PRNGKey:
    """Returns a random key specialised per sample."""
    # Note on rngs:
    # - rng_per_batch is common across replicas and accumulation steps.
    # - rng_per_microbatch is common across devices for one accumulation step.
    # - rng_per_example is specialised per sample (for independent randonmness).
    rng_per_microbatch = jax.random.fold_in(rng_per_batch, accumulation_step)
    rng_per_example = jax.random.fold_in(
        rng_per_microbatch, self._device_layout.replica_index)
    return rng_per_example

  def add_noise_to_grads(
      self,
      grads: typing.ParamsT,
      rng_per_batch: chex.PRNGKey,
      total_batch_size: int,
  ) -> tuple[typing.ParamsT, chex.Numeric]:
    """Adds noise to gradients.

    Args:
      grads: gradients to privatize.
      rng_per_batch: random number generation key.
      total_batch_size: total batch-size once accumulated over devices and steps
        (i.e. as seen by the optimizer performing the update).

    Returns:
      noisy_grads: gradients with the added noise.
      std: standard deviation used for the noise (for monitoring purposes).
    """
    return optim.add_noise_to_grads(
        grads=grads,
        rng_per_batch=self._rng_per_param_fn(rng_per_batch),
        total_batch_size=total_batch_size,
        clipping_norm=self._clipping_norm,
        rescale_to_unit_norm=self._rescale_to_unit_norm,
        noise_multiplier=self._noise_multiplier,
    )

  def l2_loss(self, params: typing.ParamsT) -> chex.Numeric:
    """Computes the squared L2 loss.

    Args:
      params: model parameters for which the loss should be computed, assumed to
        be in haiku-like format.

    Returns:
      Squared L2 loss.
    """

    return 0.5 * jnp.square(self._global_norm_fn(params))
