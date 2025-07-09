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
  state, step_on_host = updater.init(
      rng=rng,
      inputs=inputs,
  )

  # Apply update (pmapped).
  state, metrics, step_on_host = updater.update(
      state=state,
      inputs_producer=inputs_producer,
      step_on_host=step_on_host,
  )
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Generic

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import metrics_accumulator
from jax_privacy.dp_sgd import optim
from jax_privacy.dp_sgd import typing
from jax_privacy.training import averaging
from jax_privacy.training import batching as batching_module
from jax_privacy.training import devices
from jax_privacy.training import experiment_config
from jax_privacy.training import forward
from jax_privacy.training import optimizer_config as opt_config
from jax_privacy.training import updater
import optax


class Updater(
    updater.AbstractUpdater,
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    Generic[typing.InputsT, typing.ParamsT, typing.ModelStateT],
    # pytype: enable=invalid-annotation
):
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      batch_size_config: experiment_config.BatchSizeTrainConfig,
      forward_fn: forward.ForwardFn,
      grad_computer: gradients.GradientComputer,
      averaging_configs: Mapping[str, averaging.AveragingConfig] | None = None,
      weight_decay: chex.Numeric | None,
      is_trainable: experiment_config.FilterFn = lambda *args: True,
      optimizer_config: opt_config.OptimizerConfig,
      max_num_updates: int,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
      logging_config: experiment_config.LoggingConfig,
      rng: chex.PRNGKey,
      num_training_samples: int,
  ):
    """Initializes the updater.

    Args:
      batch_size_config: Configuration for the training batch-size.
      forward_fn: forward pass providing the loss function and metrics.
      grad_computer: A gradient computer for computing privatized gradients.
      averaging_configs: Configurations for the moving average of the
        parameters, indexed by name.
      weight_decay: whether to apply weight-decay on the parameters of the model
        (mechanism not privatized since it is data-independent).
      is_trainable: function to be called on each
        `(layer_name, parameter_name parameter_value)` to
        determine whether the parameter should be updated during training.
      optimizer_config: Optimizer configuration.
      max_num_updates: Maximal number of updates to perform.
      device_layout: Common args to `pmap` and `psum` for data parallelism.
      logging_config: configuration of the logging options.
      rng: random key generator; this will override the rng provided by jaxline.
      num_training_samples: number of training samples.
    """

    self._batching = batching_module.VirtualBatching(
        batch_size_init=batch_size_config.total,
        batch_size_per_device_per_step=(
            batch_size_config.per_device_per_step),
        scale_schedule=batch_size_config.scale_schedule,
    )
    self._forward_fn = forward_fn

    self._grad_computer = grad_computer
    self._weight_decay = weight_decay
    self._is_trainable = is_trainable

    self._lr_decay_schedule_fn = optimizer_config.make_lr_schedule_fn(
        max_num_updates)
    self._optimizer = optimizer_config.make_optimizer(max_num_updates)

    self._logging_config = logging_config
    self._averaging_configs: Mapping[
        str, averaging.AveragingConfig] = averaging_configs or {}
    self._num_training_samples = num_training_samples

    # This fixed key will be used instead of a per-update rng provided by the
    # pipeline, because we need to use the same key for all microbatches
    # in a full batch.
    # We will fold in the full step number at each update step.
    self._rng = rng

    self._device_layout = device_layout
    self._pmapped_init = jax.pmap(
        self._single_device_init,
        in_axes=(None, 0),  # rng, inputs
        donate_argnums=(0,),
        **device_layout.pmap_kwargs,
    )
    self._pmapped_loss_and_clipped_grad = jax.pmap(
        self._single_device_loss_and_clipped_grad,
        # args: state, inputs, accumulation_step,
        in_axes=(0, 0, None),
        donate_argnums=(1, 2),
        **device_layout.pmap_kwargs,
    )
    self._pmapped_opt_update = jax.pmap(
        self._single_device_opt_update,
        # args: state, loss, grads, metrics, batch_size
        in_axes=(0, 0, 0, 0, None),
        donate_argnums=(0, 1, 2, 3),
        **device_layout.pmap_kwargs,
    )
    self._pmapped_accumulate_grad = jax.pmap(
        self._single_device_accumulate_grads,
        # args: grads_acc, grads, every_k
        in_axes=(0, 0, None),
        donate_argnums=(0, 1),
        **device_layout.pmap_kwargs,
    )

  def init(
      self,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[updater.UpdaterState, updater.StepOnHost]:
    """Initialises parameters."""
    state = self._pmapped_init(rng, inputs)
    step_on_host = 0
    return state, step_on_host

  def update(
      self,
      state: updater.UpdaterState,
      inputs_producer: Callable[[], typing.InputsT],
      step_on_host: updater.StepOnHost,
  ) -> tuple[updater.UpdaterState, typing.Metrics, updater.StepOnHost]:
    """Updates parameters."""
    # Step 1: compute the clipped gradients, potentially accumulated over
    # multiple batches (to fit on accelerator memory).
    (loss, (network_state, metrics)), clipped_grads = (
        self._accumulate_loss_and_clipped_gradients(
            state=state,
            inputs_producer=inputs_producer,
            step_on_host=step_on_host,
        )
    )
    # Step 2: apply the optimizer update (including the noise addition) with the
    # clipped gradients computed above, and return the new state.
    state = dataclasses.replace(state, network_state=network_state)
    batch_size = jnp.asarray(self._batching.batch_size(step_on_host))
    new_state, new_metrics = self._pmapped_opt_update(
        state, loss, clipped_grads, metrics, batch_size
    )
    new_step_on_host = step_on_host + 1
    return new_state, new_metrics, new_step_on_host

  def _single_device_init(
      self,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> updater.UpdaterState:
    """Initialization function (to be pmapped)."""
    params, network_state = self._forward_fn.train_init(rng, inputs)

    trainable_params, frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    self._logging_config.maybe_log_param_shapes(
        trainable_params, prefix='[Trainable params] ')
    self._logging_config.maybe_log_param_shapes(
        frozen_params, prefix='[Frozen params] ')

    opt_state = self._optimizer.init(trainable_params)
    params_avg = {name: trainable_params for name in self._averaging_configs}
    update_step = jnp.zeros((), dtype=jnp.int32)
    noise_state = self._grad_computer.init_noise_state(trainable_params)

    return updater.UpdaterState(
        params=params,
        network_state=network_state,
        opt_state=opt_state,
        params_avg=params_avg,
        update_step=update_step,
        noise_state=noise_state,
    )

  def _single_device_loss_and_clipped_grad(
      self,
      state: updater.UpdaterState,
      inputs: typing.InputsT,
      accumulation_step: jax.Array,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    """Updates parameters (to be pmapped)."""
    # Note on rngs:
    # - rng_per_batch is common across replicas and accumulation steps.
    # - rng_per_microbatch is common across devices for one accumulation step.
    # - rng_per_local_microbatch is suitable for per sample randomness.
    rng_per_batch = jax.random.fold_in(self._rng, state.update_step)
    rng_per_microbatch = jax.random.fold_in(rng_per_batch, accumulation_step)
    rng_per_local_microbatch = jax.random.fold_in(
        rng_per_microbatch, self._device_layout.replica_index)

    # Potentially split params between trainable parameters and frozen
    # parameters. Trainable parameters get updated, while frozen parameters do
    # not.
    trainable_params, frozen_params = hk.data_structures.partition(
        self._is_trainable, state.params)
    def loss_fn(train_params, *args):
      all_params = hk.data_structures.merge(train_params, frozen_params)
      return self._forward_fn.train_forward(all_params, *args)

    # Compute the clipped gradients, on this device.
    (loss, (network_state, metrics)), device_grads = (
        self._grad_computer.loss_and_clipped_gradients(
            loss_fn=loss_fn,
            params=trainable_params,
            network_state=state.network_state,
            rng_per_local_microbatch=rng_per_local_microbatch,
            inputs=inputs,
        )
    )

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

  def _single_device_accumulate_grads(
      self,
      grads_acc: typing.ParamsT | None,
      grads: typing.ParamsT,
      every_k: jax.Array,
  ) -> typing.ParamsT:
    """Returns accumulated gradients (to be pmapped)."""
    every_k_inv = 1.0 / every_k
    if grads_acc is None:
      return jax.tree_util.tree_map(lambda x: x*every_k_inv, grads)
    else:
      return jax.tree_util.tree_map(
          lambda acc, x: acc + x*every_k_inv,
          grads_acc,
          grads,
      )

  def _accumulate_loss_and_clipped_gradients(
      self,
      *,
      state: updater.UpdaterState,
      inputs_producer: Callable[[], typing.InputsT],
      step_on_host: updater.StepOnHost,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    """Accumulates clipped gradients."""
    every_k = int(self._batching.apply_update_every(step_on_host))
    total_batch_size = self._batching.batch_size(step_on_host)

    # Accumulate over every_k steps.
    # Accumulate on axis 1 since 0 is used for data parallelism.
    metrics_acc = metrics_accumulator.MetricsAccumulator()
    grads_acc = None
    network_state = None
    loss_acc = 0.0
    local_batch_size = total_batch_size // every_k
    for accumulation_step in jnp.arange(0, every_k, dtype=jnp.int32):
      inputs = inputs_producer()
      # Compute loss and grad over current batch.
      (loss, (network_state, metrics)), grads = (
          self._pmapped_loss_and_clipped_grad(state, inputs, accumulation_step))
      # Accumulate loss.
      loss_acc += loss / every_k
      # Accumulate metrics.
      metrics_acc = metrics_acc.accumulate(
          other=metrics, other_count=local_batch_size)
      # Accumulate grads.
      grads_acc = self._pmapped_accumulate_grad(grads_acc, grads, every_k)

    metrics = metrics_acc.to_metrics()
    return (loss_acc, (network_state, metrics)), grads_acc

  # We need to disable the too-many-positional-arguments check because this
  # method will be pmapped and pmapped functions cannot have kwargs.
  def _single_device_opt_update(  # pylint: disable=too-many-positional-arguments
      self,
      state: updater.UpdaterState,
      loss: jax.Array,
      clipped_grads: typing.ParamsT,
      metrics: typing.Metrics,
      total_batch_size: jax.Array,
  ) -> tuple[updater.UpdaterState, typing.Metrics]:
    """Updates the state with `grads`."""
    rng_per_batch = jax.random.fold_in(self._rng, state.update_step)

    # Add noise to the gradients.
    grads, std, new_noise_state = self._grad_computer.add_noise_to_grads(
        clipped_grads,
        rng_per_batch,
        total_batch_size,
        state.noise_state,
    )

    trainable_params, frozen_params = hk.data_structures.partition(
        self._is_trainable, state.params
    )
    learning_rate = self._lr_decay_schedule_fn(state.update_step)

    # Perform the update on the model parameters.
    updates, opt_state = self._optimizer.update(
        grads, state.opt_state, trainable_params
    )
    trainable_params = optax.apply_updates(trainable_params, updates)

    # Regularization
    l2_loss = self._grad_computer.l2_loss(trainable_params)
    regularization_loss = self._weight_decay * l2_loss

    # Manually apply weight decay with the current learning-rate.
    trainable_params = optim.apply_weight_decay(
        trainable_params,
        learning_rate=learning_rate,
        weight_decay=self._weight_decay,
    )

    # Perform parameter averaging.
    params_avg = {
        name: config.make()(
            state.params_avg[name], trainable_params, state.update_step)
        for name, config in self._averaging_configs.items()
    }

    # Merge the updated parameters with the parameters that are supposed to
    # remain frozen during training.
    params = hk.data_structures.merge(trainable_params, frozen_params)

    new_update_step = optax.safe_int32_increment(state.update_step)

    new_state = updater.UpdaterState(
        params=params,
        params_avg=params_avg,
        network_state=state.network_state,
        opt_state=opt_state,
        update_step=new_update_step,
        noise_state=new_noise_state,
    )

    # Log all relevant statistics in a dictionary.
    # NOTE: approximation below if the batch-size is not constant.
    data_seen = total_batch_size * new_update_step

    scalars_last = {
        'reg': regularization_loss,
        'batch_size': total_batch_size,
        'update_every': self._batching.apply_update_every(new_update_step),
        'l2_loss': l2_loss,
        'update_step': new_update_step,  # use value after opt update
        'learning_rate': learning_rate,
        'data_seen': data_seen,
        'epochs': data_seen / self._num_training_samples,
        'std': std,
        'obj': regularization_loss + loss,
        'grads_norm': self._grad_computer.global_norm(grads),
        **self._compute_gradient_stats(
            rng_per_batch=rng_per_batch,
            avg_grads=grads,
            grad_norms_per_sample=metrics.per_example.get('grad_norm'),
            total_batch_size=total_batch_size,
            # Pass the previous noise state as it was the state used to create
            # the noise for the current update.
            noise_state=state.noise_state,
        ),
    }

    scalars_last.update(
        self._logging_config.additional_training_metrics(params, grads)
    )

    new_metrics = typing.Metrics(
        scalars_last={**metrics.scalars_last, **scalars_last},
        scalars_sum={**metrics.scalars_sum},
        scalars_avg={**metrics.scalars_avg},
        per_example={**metrics.per_example},
    )

    return new_state, new_metrics

  def _compute_gradient_stats(
      self,
      *,
      total_batch_size: jax.Array,
      rng_per_batch: chex.PRNGKey,
      avg_grads: typing.ParamsT,
      grad_norms_per_sample: chex.ArrayBatched,
      noise_state: typing.NoiseStateT,
  ) -> Mapping[str, chex.Numeric]:
    """Compute various gradient statistics for logging."""
    stats = {}

    # Log Signal-to-Noise Ratio.
    if self._logging_config.snr_global or self._logging_config.snr_per_layer:
      noise = self._recompute_noise(
          total_batch_size=total_batch_size,
          grads_like=avg_grads,
          rng_per_batch=rng_per_batch,
          noise_state=noise_state,
      )
      def snr(s, n):
        return (self._grad_computer.global_norm(s) /
                self._grad_computer.global_norm(n))
      if self._logging_config.snr_global:
        stats['snr_global'] = snr(avg_grads, noise)
      if self._logging_config.snr_per_layer:
        snr_per_layer = jax.tree_util.tree_map(snr, avg_grads, noise)
        stats.update({
            f'snr_{mod_name}_{name}': value
            for mod_name, name, value in hk.data_structures.traverse(
                snr_per_layer
            )
        })

    if self._logging_config.grad_clipping:
      if not self._grad_computer.using_clipped_grads:
        stats.update(grads_clipped=0.0)
      else:
        grads_clipped = jnp.mean(
            jnp.greater(grad_norms_per_sample,
                        self._grad_computer.clipping_norm))
        stats.update(
            grads_clipped=grads_clipped,
            grad_norms_before_clipping_mean=jnp.mean(grad_norms_per_sample),
            grad_norms_before_clipping_median=jnp.median(grad_norms_per_sample),
            grad_norms_before_clipping_min=jnp.min(grad_norms_per_sample),
            grad_norms_before_clipping_max=jnp.max(grad_norms_per_sample),
            grad_norms_before_clipping_std=jnp.std(grad_norms_per_sample),
        )

    return stats

  def _recompute_noise(
      self,
      total_batch_size: jax.Array,
      grads_like: typing.ParamsT,
      rng_per_batch: chex.PRNGKey,
      noise_state: typing.NoiseStateT,
  ) -> typing.ParamsT:
    """Re-create the noise with the same RNG and add it to zeros."""
    noise, unused_std, unused_noise_state = (
        self._grad_computer.add_noise_to_grads(
            total_batch_size=total_batch_size,
            grads=jax.tree_util.tree_map(jnp.zeros_like, grads_like),
            rng_per_batch=rng_per_batch,
            noise_state=noise_state,
        )
    )
    return noise
