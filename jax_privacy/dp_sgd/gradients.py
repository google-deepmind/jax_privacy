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

"""Differentially private gradient computation."""

import abc
from collections.abc import Mapping
import dataclasses
import functools
from typing import Callable, Generic

import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import grad_clipping_utils as gcu
from jax_privacy.dp_sgd import optim
from jax_privacy.dp_sgd import typing
import optax
from typing_extensions import override


class GradientComputer(
    # False positive error with pytype failing to use a `TypeVar` imported
    # from elsewhere.
    # pytype: disable=invalid-annotation
    abc.ABC,
    Generic[
        typing.InputsT, typing.ParamsT, typing.ModelStateT, typing.NoiseStateT
    ],
    # pytype: enable=invalid-annotation
):
  """Computes (potentially) clipped and noisy gradients."""

  def __init__(
      self,
      *,
      clipping_norm: float | None,
      noise_multiplier: float | None,
      rescale_to_unit_norm: bool,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      rng_per_param_fn: Callable[[chex.PRNGKey], chex.PRNGKey] = lambda x: x,
      global_norm_fn: typing.NormFn = optax.global_norm,
  ):
    """Initialises the gradient computation.

    Args:
      clipping_norm: maximum L2 norm to which the input tree should be clipped.
      noise_multiplier: standard deviation of the noise to add to the average of
        the clipped gradient to make it differentially private. It will be
        multiplied by `clipping_norm / total_batch_size` before the noise gets
        actually added.
      rescale_to_unit_norm: If true, additionally rescale the clipped gradient
        by 1/clipping_norm so it has an L2 norm of at most one.
      per_example_grad_method: Per-example gradient clipping method to use. Does
        not affect the results, but controls speed/memory trade-off.
      rng_per_param_fn: Optional callable to allow gradient noise random keys to
        be specialised for different param slices.
      global_norm_fn: function to compute the L2 norm of an ArrayTree.
    """
    self._clipping_norm = clipping_norm
    self._noise_multiplier = noise_multiplier
    self._rescale_to_unit_norm = rescale_to_unit_norm
    self._per_example_grad_method = per_example_grad_method
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
      *,
      loss_fn: typing.LossFn,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_local_microbatch: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.ParamsT:
    """Computes unclipped gradients of the given loss function.

    Args:
      loss_fn: Loss function whose gradients are required.
      params: Trainable parameters.
      network_state: Network state input to `loss_fn`.
      rng_per_local_microbatch: Random number key for the batch. The caller must
        provide independent keys for different training steps, accumulation
        steps (if using microbatching), and model replicas (if invoked in a
        pmap).
      inputs: Inputs to `loss_fn`.

    Returns:
      Unclipped gradients.
    """
    # Compute gradients of the loss function w.r.t. the parameters.
    device_grads, unused_aux = jax.grad(loss_fn, has_aux=True)(
        params, network_state, rng_per_local_microbatch, inputs
    )
    return device_grads

  def loss_and_clipped_gradients(
      self,
      *,
      loss_fn: typing.LossFn,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_local_microbatch: chex.PRNGKey,
      inputs: typing.InputsT,
      state_acc_strategies: gcu.StateAccumulationStrategyTree = gcu.Reject(),
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    """Computes (potentially) clipped gradients of the given loss function.

    Args:
      loss_fn: Loss function whose clipped gradients are required.
      params: Trainable parameters.
      network_state: Network state input to `loss_fn`.
      rng_per_local_microbatch: Random number key for the batch. The caller must
        provide independent keys for different training steps, accumulation
        steps (if using microbatching), and model replicas (if invoked in a
        pmap).
      inputs: Inputs to `loss_fn`.
      state_acc_strategies: Prefix tree of network state accumulation
        strategies.

    Returns:
      Tuple consisting of (loss-and-aux, clipped_grads)
      where `loss-and-aux` is as is returned by `loss_fn` (with the addition
      of the grad norm per example in the metrics).
    """
    # Compute clipped-per-example gradients of the loss function w.r.t. the
    # parameters.
    value_and_clipped_grad_fn = self.value_and_clipped_grad(
        jax.value_and_grad(loss_fn, has_aux=True),
        state_acc_strategies=state_acc_strategies,
    )
    return value_and_clipped_grad_fn(
        params, network_state, rng_per_local_microbatch, inputs
    )

  def value_and_clipped_grad(
      self,
      value_and_grad_fn: typing.ValueAndGradFn,
      *,
      state_acc_strategies: gcu.StateAccumulationStrategyTree = gcu.Reject(),
  ) -> typing.ValueAndGradFn:
    """Creates the function computing (potentially) clipped gradients.

    Args:
      value_and_grad_fn: Function that produces unclipped gradients. It is
        expected to have the following signature: `(loss, aux), grad =
        grad_fn(params, network_state, rng_key, inputs)`.
      state_acc_strategies: Prefix tree of network state accumulation
        strategies. The default is to raise an error if any network state is
        present, but this can be overridden, e.g. to average state across
        microbatches. CAUTION - Any approach in which the state depends on the
        inputs _and_ influences trainable parameters (as will be the case with
        batch normalisation) will invalidate the DP guarantees, as it's
        bypassing the DP noise/clipping.

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

    return self._per_example_grad_method(
        value_and_grad_fn,
        clipping_fn=clipping_fn,
        state_acc_strategies=state_acc_strategies,
    )

  @abc.abstractmethod
  def init_noise_state(self, params: typing.ParamsT) -> typing.NoiseStateT:
    """Returns a new noise_state to be used for adding noise to gradients."""

  @abc.abstractmethod
  def add_noise_to_grads(
      self,
      grads: typing.ParamsT,
      rng_per_batch: chex.PRNGKey,
      total_batch_size: jax.Array,
      noise_state: typing.NoiseStateT,
  ) -> tuple[typing.ParamsT, jax.Array, typing.NoiseStateT]:
    """Adds noise to gradients.

    Args:
      grads: gradients to privatize.
      rng_per_batch: random number generation key.
      total_batch_size: total batch-size once accumulated over devices and steps
        (i.e. as seen by the optimizer performing the update).
      noise_state: additional state required to compute noise.

    Returns:
      noisy_grads: gradients with the added noise.
      std: standard deviation used for the noise (for monitoring purposes).
      noise_state: (updated, if needed) state required to compute noise.
    """

  def l2_loss(self, params: typing.ParamsT) -> chex.Numeric:
    """Computes the squared L2 loss.

    Args:
      params: model parameters for which the loss should be computed, assumed to
        be in haiku-like format.

    Returns:
      Squared L2 loss.
    """

    return 0.5 * jnp.square(self._global_norm_fn(params))


class DpsgdGradientComputer(
    # pytype: disable=not-indexable
    GradientComputer[
        typing.InputsT,
        typing.ParamsT,
        typing.ModelStateT,
        Mapping[str, jax.Array],
    ]
    # pytype: enable=not-indexable
):
  """Gradient computer for DP-SGD."""

  def init_noise_state(self, params: typing.ParamsT) -> Mapping[str, jax.Array]:
    """Initialize noise state for DP-SGD."""
    del params
    return {}

  @override
  def add_noise_to_grads(
      self,
      grads: typing.ParamsT,
      rng_per_batch: chex.PRNGKey,
      total_batch_size: jax.Array,
      noise_state: Mapping[str, jax.Array],
  ) -> tuple[typing.ParamsT, jax.Array, Mapping[str, jax.Array]]:
    noisy_grads, std = optim.add_noise_to_grads(
        grads=grads,
        rng_per_batch=self._rng_per_param_fn(rng_per_batch),
        total_batch_size=total_batch_size,
        clipping_norm=self._clipping_norm,
        rescale_to_unit_norm=self._rescale_to_unit_norm,
        noise_multiplier=self._noise_multiplier,
    )
    return noisy_grads, std, noise_state


@chex.dataclass(frozen=True, mappable_dataclass=True)
class DpftrlNoiseState:
  """State for DP-FTRL."""

  rngs: typing.SquareMatrix
  iteration: jnp.int64 = dataclasses.field(default_factory=lambda: jnp.int64(0))  # pylint: disable=invalid-field-call

  def __getitem__(self, item):
    return getattr(self, item)

  def __iter__(self):
    for field in dataclasses.fields(self):
      yield getattr(self, field.name)

  def __len__(self):
    return len(dataclasses.fields(self))


def check_is_matrix(tensor: jax.Array):
  if tensor.ndim != 2:
    raise ValueError(
        f'`tensor` should be a matrix of rank 2. Got shape {tensor.shape}'
    )


def check_square(matrix: jax.Array):
  if matrix.shape[0] != matrix.shape[1]:
    raise ValueError(f'`matrix` should be square, found\n{matrix}')


def check_lower_triangular(matrix: jax.Array, **allclose_kwargs):
  if not jnp.allclose(matrix, jnp.tril(matrix), **allclose_kwargs):
    raise ValueError(f'`matrix` should be lower-triangular, found\n{matrix}')


class DpftrlGradientComputer(
    # pytype: disable=not-indexable
    GradientComputer[
        typing.InputsT,
        typing.ParamsT,
        typing.ModelStateT,
        DpftrlNoiseState,
    ]
    # pytype: enable=not-indexable
):
  """Gradient computer for DP-FTRL via matrix factorization."""

  def __init__(
      self,
      *,
      correlation_matrix: typing.SquareMatrix,
      correlation_unroll: int | bool | None = None,
      **kwargs,
  ):
    """Initialises the DPFTRL gradient computation.

    Args:
      correlation_matrix: `C^{-1}` correlation matrix used to compute the noise.
        Must be a lower triangular matrix.
      correlation_unroll: Controls the `unroll` parameter for the fori_loop used
        to generate correlated noise. This could improve speed of generation but
        may come at some cost in memory.
      **kwargs: keyword arguments to pass to the parent class.
    """
    check_is_matrix(correlation_matrix)
    check_square(correlation_matrix)
    check_lower_triangular(correlation_matrix)

    super().__init__(**kwargs)
    self._correlation_matrix = correlation_matrix
    self._correlation_unroll = correlation_unroll

  def init_noise_state(self, params: typing.ParamsT) -> DpftrlNoiseState:
    """Initialize noise state for DP-FTRL."""
    del params
    return DpftrlNoiseState(
        iteration=jnp.int64(0),
        # 2 is the shape for `chex.PRNGKey`s.
        rngs=jnp.empty((len(self._correlation_matrix), 2), dtype=jnp.uint32),
    )

  def _check_iteration_valid(self, iteration: jax.Array):
    if iteration >= len(self._correlation_matrix):
      raise ValueError(
          f'Iteration {iteration} goes beyond number of rows in correlation'
          f' matrix {len(self._correlation_matrix)}'
      )

  def _correlated_noisy_grads(
      self,
      curr_step: jax.Array,
      noisy_grads: jax.Array,
      *,
      correlation_vector: jax.Array,
      rngs: jax.Array,
      noise_std: chex.Numeric,
  ) -> jax.Array:
    """Performs one step of noise correlation on gradients for DP-FTRL updates.

    This function serves to perform the core noise correlation step in DP-FTRL.
    In particular, given a set of prior RNGs `rngs`, and weights for the noise
    generated by each rng `correlation_vector`, this generates the noise for a
    given rng as indicated by `curr_step`. Importantly, this performs only one
    step of the correlation, as is intended to be called inside a loop to
    perform all correlation steps. Lastly, this method is designed to directly
    add noise to the gradients (rather than to first generate the correlated
    noise) so as to reduce memory overhead.

    This function performs no error checking on the input `rngs` and
    `correlation_vector` and assumes that the caller has handled this.

    Args:
      curr_step: The idx to take from `correlation_vector` and `rngs`.
      noisy_grads: The current noisy gradients for the model update. This is
        iteratively updated upon successive calls to this function.
      correlation_vector: The weights associated with each rng.
      rngs: the seeds from which to draw noise. In DP-FTRL terms, which columns
        of the `correlation_matrix` to draw noise from.
      noise_std: the standard deviation of isotropic Gaussian noise to use.
        Corresponds to scaling the noise_multiplier by the clipping norm.

    Returns:
      The noisy gradients, where noise is correlated as per DP-FTRL.
    """

    def correlated_noisy_grads():
      rng = rngs[curr_step]
      return optim.tree_map_add_normal_noise(
          noisy_grads, weight * noise_std, self._rng_per_param_fn(rng)
      )

    weight = correlation_vector[curr_step]
    return jax.lax.cond(weight, correlated_noisy_grads, lambda: noisy_grads)

  @override
  def add_noise_to_grads(
      self,
      grads: typing.ParamsT,
      rng_per_batch: chex.PRNGKey,
      total_batch_size: jax.Array,
      noise_state: DpftrlNoiseState,
  ) -> tuple[typing.ParamsT, jax.Array, DpftrlNoiseState]:
    iteration = noise_state['iteration']
    jax.debug.callback(self._check_iteration_valid, iteration)
    rngs = noise_state['rngs']
    rngs = rngs.at[iteration].set(rng_per_batch)

    correlation_vector = self._correlation_matrix[iteration]
    lower, upper = 0, correlation_vector.shape[0]

    clipping_scale = optim.compute_clipping_norm_scale(
        self._clipping_norm, self._rescale_to_unit_norm, total_batch_size
    )
    noise_std = optim.compute_noise_std(self._noise_multiplier, clipping_scale)

    correlation_fn = functools.partial(
        self._correlated_noisy_grads,
        correlation_vector=correlation_vector,
        rngs=rngs,
        noise_std=noise_std,
    )

    if self._noise_multiplier is not None and self._noise_multiplier:
      noisy_grads = jax.lax.fori_loop(
          lower, upper, correlation_fn, grads, unroll=self._correlation_unroll
      )
      noise_std *= jnp.sqrt(jnp.sum(jnp.square(correlation_vector)))
    else:
      noisy_grads = grads

    new_state = DpftrlNoiseState(
        iteration=noise_state['iteration'] + 1, rngs=rngs
    )
    return noisy_grads, noise_std, new_state
