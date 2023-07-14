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
  # The updater requires a forward function, specification of virtual batching,
  # and a DP-SGD gradient computer:
  updater = dp_updater.Updater(
        batching=batching,  # see `batching.py`
        forward_fn=forward_fn,  # see `forward.py`
        grad_computer=grad_computer,  # see `gradients.py`
  )

  ...

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

import functools
from typing import Mapping, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import batching as batching_module
from jax_privacy.src.dp_sgd import devices
from jax_privacy.src.dp_sgd import gradients
from jax_privacy.src.dp_sgd import optim
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import averaging
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import forward
from jax_privacy.src.training import optimizer_config as opt_config
from jax_privacy.src.training import updater
from jaxline import utils as jaxline_utils
import optax


class Updater(updater.AbstractUpdater):
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      batching: batching_module.VirtualBatching,
      forward_fn: forward.ForwardFn,
      grad_computer: gradients.GradientComputer,
      weight_decay: Optional[chex.Numeric],
      is_trainable: experiment_config.FilterFn = lambda *args: True,
      optimizer_config: opt_config.OptimizerConfig,
      max_num_updates: int,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
      logging_config: experiment_config.LoggingConfig,
      rng_seed: int = 42,
  ):
    """Initializes the updater.

    Args:
      batching: virtual batching that allows to use 'virtual' batches across
        devices and steps.
      forward_fn: forward pass providing the loss function and metrics.
      grad_computer: Computer of the gradient.
      weight_decay: whether to apply weight-decay on the parameters of the model
        (mechanism not privatized since it is data-independent).
      is_trainable: function to be called on each
        `(layer_name, parameter_name parameter_value)` to
        determine whether the parameter should be updated during training.
      optimizer_config: Optimizer configuration.
      max_num_updates: Maximal number of updates to perform.
      device_layout: Common args to `pmap` and `psum` for data parallelism.
      logging_config: configuration of the logging options.
      rng_seed: seed to use for the random key generator;
        this will override the rng provided by jaxline.
    """
    self._batching = batching
    self._forward_fn = forward_fn

    self._grad_computer = grad_computer
    self._weight_decay = weight_decay
    self._is_trainable = is_trainable

    # Create an optimizer that will only apply the update every
    # `k=batching.apply_update_every` steps, and accumulate gradients
    # in-between so that we can use a large 'virtual' batch-size.
    # For example, if `k` is 4, on the first three steps, the optimizer will
    # store the gradients of the mini-batches and not perform any update.
    # On the fourth step, the optimizer will average the gradients of the fourth
    # mini-batch with those of the first three, perform the update, and reset
    # its memory, and so on. This allows to use large virtual batch-sizes.
    self._lr_decay_schedule_fn = optimizer_config.make_lr_schedule_fn(
        max_num_updates)
    self._optimizer = optax.MultiSteps(
        optimizer_config.make_optimizer(max_num_updates),
        batching.apply_update_every,
    )

    self._logging_config = logging_config

    # This key will be used instead of the rng provided by jaxline
    # since the latter is updated at every step.
    self._rng_init = jax.random.PRNGKey(rng_seed)

    self._device_layout = device_layout
    self._pmapped_init = jax.pmap(
        self._single_device_init,
        in_axes=(None, 0),  # rng, inputs
        donate_argnums=(0,),
        **device_layout.pmap_kwargs)
    self._pmapped_update = jax.pmap(
        self._single_device_update,
        in_axes=(0, 0, 0, 0),  # params, net_state, opt_state, inputs
        donate_argnums=(0, 1, 2, 4),
        **device_layout.pmap_kwargs)
    self._pmapped_evaluate = jax.pmap(
        self._single_device_evaluate,
        in_axes=(0, 0, None, 0),  # params, net_state, rng, inputs
        donate_argnums=(2,),
        **device_layout.pmap_kwargs)

    self._init_average = jax.pmap(
        lambda x: x,
        **device_layout.pmap_kwargs)
    self._average_ema = jax.pmap(
        averaging.ema,
        in_axes=(0, 0, None, None),  # old_average, new, mu, start_step
        donate_argnums=(0,),
        **device_layout.pmap_kwargs)
    self._average_polyak = jax.pmap(
        averaging.polyak,
        in_axes=(0, 0, None),  # old_average, new, start_step
        donate_argnums=(0,),
        **device_layout.pmap_kwargs)

  def step_count_from_opt_state(
      self,
      opt_state: optax.MultiStepsState,
  ) -> updater.StepCount:
    """Returns the hierarchical step number."""
    assert isinstance(opt_state, optax.MultiStepsState)
    return updater.StepCount(
        update_step=int(jaxline_utils.get_first(opt_state.gradient_step)),
        accumulation_step=int(jaxline_utils.get_first(opt_state.mini_step)),
    )

  def _regularization(
      self,
      params: typing.ParamsT,
  ) -> tuple[chex.Numeric, chex.Numeric]:
    l2_loss = self._grad_computer.l2_loss(params)
    return self._weight_decay * l2_loss, l2_loss

  def init(
      self,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT, optax.MultiStepsState,
             updater.StepCount]:
    """Initialises parameters."""
    params, network_state, opt_state = self._pmapped_init(rng, inputs)

    # Non-vectorised Python integers, so keep outside the pmap.
    step_count = updater.StepCount(
        update_step=0,
        accumulation_step=0,
    )

    return params, network_state, opt_state, step_count

  def update(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      opt_state: optax.MultiStepsState,
      step_count: updater.StepCount,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT, optax.MultiStepsState,
             updater.StepCount, typing.Metrics]:
    """Updates parameters."""
    # The function below is p-mapped, so arguments must be provided without name
    # and in the right order.
    params, network_state, opt_state, metrics = self._pmapped_update(
        params, network_state, opt_state, inputs)

    # Replicate the logic in optax.MultiSteps to determine the updated
    # hierarchical step (full + inner) in Python integers. This makes it
    # available to the caller without blocking on the device computation.
    every_k = self._batching.apply_update_every(step_count.update_step)
    step_count = step_count.next(every_k)

    return params, network_state, opt_state, step_count, metrics

  def evaluate(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.Metrics:
    """Evaluates model parameters."""
    # The function below is p-mapped, so arguments must be provided without name
    # and in the right order.
    return jaxline_utils.get_first(
        self._pmapped_evaluate(params, network_state, rng, inputs))

  def optimizer(self) -> optax.GradientTransformation:
    return self._optimizer.gradient_transformation()

  def init_average(
      self,
      params: typing.ParamsT,
  ) -> typing.ParamsT:
    return self._init_average(params)

  def update_ema(
      self,
      ema_params: typing.ParamsT,
      params: typing.ParamsT,
      opt_state: optax.MultiStepsState,
      *,
      mu: chex.Numeric,
      start_step: chex.Numeric,
  ) -> typing.ParamsT:
    # We only perform parameter averaging if the current step corresponds to an
    # update step (and not a gradient accumulation step).
    if self._optimizer.has_updated(jaxline_utils.get_first(opt_state)):
      t = jaxline_utils.get_first(opt_state.gradient_step) - start_step
      return self._average_ema(ema_params, params, mu, t)
    else:
      return ema_params

  def update_polyak(
      self,
      polyak_params: typing.ParamsT,
      params: typing.ParamsT,
      opt_state: optax.MultiStepsState,
      *,
      start_step: chex.Numeric,
  ) -> typing.ParamsT:
    # We only perform parameter averaging if the current step corresponds to an
    # update step (and not a gradient accumulation step).
    if self._optimizer.has_updated(jaxline_utils.get_first(opt_state)):
      t = jaxline_utils.get_first(opt_state.gradient_step) - start_step
      return self._average_polyak(polyak_params, params, t)
    else:
      return polyak_params

  def _single_device_init(
      self,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT, optax.MultiStepsState]:
    """Initialization function (to be pmapped)."""
    params, network_state = self._forward_fn.train_init(rng, inputs)

    trainable_params, unused_frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    opt_state = self._optimizer.init(trainable_params)

    return params, network_state, opt_state

  def _single_device_update(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      opt_state: optax.MultiStepsState,
      inputs: typing.InputsT,
  ) -> tuple[
      typing.ParamsT, typing.ModelStateT, optax.MultiStepsState,
      typing.Metrics,
  ]:
    """Updates parameters (to be pmapped)."""
    # Potentially split params between trainable parameters and frozen
    # parameters. Trainable parameters get updated, while frozen parameters do
    # not.
    trainable_params, frozen_params = hk.data_structures.partition(
        self._is_trainable, params)
    def loss_fn(train_params, *args):
      all_params = hk.data_structures.merge(train_params, frozen_params)
      return self._forward_fn.train_forward(all_params, *args)

    self._logging_config.maybe_log_param_shapes(
        trainable_params, prefix='[Trainable params] ')
    self._logging_config.maybe_log_param_shapes(
        frozen_params, prefix='[Frozen params] ')

    (
        trainable_params,
        network_state,
        opt_state,
        metrics,
    ) = self._update_with_stats(
        loss_fn=loss_fn,
        params=trainable_params,
        network_state=network_state,
        opt_state=opt_state,
        inputs=inputs,
    )

    # Merge the updated parameters with the parameters that are supposed to
    # remain frozen during training.
    params = hk.data_structures.merge(trainable_params, frozen_params)

    return params, network_state, opt_state, metrics

  def _update_with_stats(
      self,
      loss_fn: typing.LossFn,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      opt_state: optax.MultiStepsState,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT, optax.MultiStepsState,
             typing.Metrics]:
    """Updates parameters and computes relevant training statistics."""
    # `rng_per_batch` is common across replicas and accumulation steps.
    # NOTE: folding an int (scalar) array into a random key is valid, but fails
    # the type check, hence why pytype is disabled below.
    rng_per_batch = jax.random.fold_in(self._rng_init, opt_state.gradient_step)

    # Compute the regularization.
    reg, l2_loss = self._regularization(params)

    if self._logging_config.grad_alignment:
      # Compute (non-clipped) gradients w.r.t. trainable parameters.
      # Do so before `params` and `network_state` are updated.
      clean_grads = self._grad_computer.clean_gradients(
          loss_fn=loss_fn,
          params=params,
          network_state=network_state,
          rng_per_batch=rng_per_batch,
          accumulation_step=opt_state.mini_step,
          inputs=inputs,
      )

    # Compute the clipped gradients (across all replicas).
    (loss, (network_state, metrics)), avg_grads = (
        self._grad_computer.loss_and_clipped_gradients(
            loss_fn=loss_fn,
            params=params,
            network_state=network_state,
            rng_per_batch=rng_per_batch,
            accumulation_step=opt_state.mini_step,
            inputs=inputs,
        )
    )

    # Compute the noise scale based on `noise_multiplier`, the batch-size and
    # the clipping-norm. Compute our 'final' gradients `grads`: add the clipped
    # data-dependent gradients (`avg_grads`) and the noise to be added to
    # achieved differential privacy.
    grads, std = self._grad_computer.add_noise_to_grads(
        total_batch_size=self._batching.batch_size(opt_state.gradient_step),
        grads=avg_grads,
        rng_per_batch=rng_per_batch,
    )

    # The update step is logged in the optimizer state (by optax.MultiSteps)
    # under the name of 'gradient_step'.
    # Note that the learning rate schedule evolves with `update_step`
    # rather than `global_step`, since the former accounts for the fact that
    # gradients may be accumulated over multiple global steps.
    learning_rate = self._lr_decay_schedule_fn(opt_state.gradient_step)
    params, opt_state = self._opt_update(
        params, opt_state, grads, learning_rate)

    # Log all relevant statistics in a dictionary.
    loss_vector = metrics.per_example['loss']
    scalars = {
        'noise_std': std,
        'loss': loss,
        'loss_mean': jnp.mean(loss_vector),
        'loss_min': jnp.min(loss_vector),
        'loss_max': jnp.max(loss_vector),
        'loss_std': jnp.std(loss_vector),
        'loss_median': jnp.median(loss_vector),
        'reg': reg,
        'batch_size': self._batching.batch_size(opt_state.gradient_step),
        'update_every': self._batching.apply_update_every(
            opt_state.gradient_step),
        'l2_loss': l2_loss,
        'obj': (reg + loss),
        'grads_norm': self._grad_computer.global_norm(grads),
        'update_step': opt_state.gradient_step,  # use value after opt update
        'learning_rate': learning_rate,
    }

    scalars.update(metrics.scalars_avg)

    # Possibly log additional statistics from the gradient.
    scalars.update(self._compute_gradient_stats(
        opt_state=opt_state,
        rng_per_batch=rng_per_batch,
        avg_grads=avg_grads,
        grad_norms_per_sample=metrics.per_example.get('grad_norm'),
    ))

    if self._logging_config.grad_alignment:
      # TODO: This only computes alignment on the current shard.
      scalars.update(grad_alignment=optim.cosine_distance(grads, clean_grads))

    metrics = typing.Metrics(
        scalars_avg=scalars,
        per_example=metrics.per_example,
        scalars_sum=metrics.scalars_sum,
    )
    return params, network_state, opt_state, metrics

  def _opt_update(
      self,
      params: typing.ParamsT,
      opt_state: optax.MultiStepsState,
      grads: typing.ParamsT,
      learning_rate: chex.Array,
  ) -> tuple[typing.ParamsT, optax.MultiStepsState]:
    """Returns `params` and `opt_state` updated with `grads`."""
    # Perform the update on the model parameters (no-op if this step
    # is meant to accumulate gradients rather than performing the model update).
    updates, opt_state = self._optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Manually apply weight decay with the current learning-rate.
    if self._weight_decay:
      params = jax.lax.cond(
          self._optimizer.has_updated(opt_state),
          functools.partial(  # decay parameters if this is an update step
              optim.apply_weight_decay,
              learning_rate=learning_rate,
              weight_decay=self._weight_decay,
          ),
          lambda p: p,  # do not decay if this is an accumulation step
          params,
      )

    return params, opt_state

  def _compute_gradient_stats(
      self,
      *,
      opt_state: optax.MultiStepsState,
      rng_per_batch: chex.PRNGKey,
      avg_grads: typing.ParamsT,
      grad_norms_per_sample: chex.ArrayBatched,
  ) -> Mapping[str, chex.Numeric]:
    """Compute various gradient statistics for logging."""
    stats = {}

    # Log Signal-to-Noise Ratio.
    if self._logging_config.snr_global or self._logging_config.snr_per_layer:
      noise = self._recompute_noise(
          opt_state=opt_state,
          grads_like=avg_grads,
          rng_per_batch=rng_per_batch,
      )
      def snr(s, n):
        return (self._grad_computer.global_norm(s) /
                self._grad_computer.global_norm(n))
    if self._logging_config.snr_global:
      stats['snr_global'] = snr(avg_grads, noise)
    if self._logging_config.snr_per_layer:
      if noise is None:
        noise = self._recompute_noise(
            opt_state=opt_state,
            grads_like=avg_grads,
            rng_per_batch=rng_per_batch,
        )
      signal_to_noise_per_layer = jax.tree_map(snr, avg_grads, noise)
      stats.update({
          f'snr_{mod_name}_{name}': value
          for mod_name, name, value in hk.data_structures.traverse(
              signal_to_noise_per_layer)})

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

  def _recompute_noise(self, opt_state, grads_like, rng_per_batch):
    """Re-create the noise with the same RNG and add it to zeros."""
    noise, unused_std = self._grad_computer.add_noise_to_grads(
        total_batch_size=self._batching.batch_size(opt_state.gradient_step),
        grads=jax.tree_map(jnp.zeros_like, grads_like),
        rng_per_batch=rng_per_batch,
    )
    return noise

  def _single_device_evaluate(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.Metrics:
    """Evaluates model parameters (to be pmapped)."""
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
