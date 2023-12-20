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

"""Computing gradients that are clipped per sample."""

import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import grad_clipping_utils
from jax_privacy.src.dp_sgd import typing
import optax


def safe_div(
    numerator: chex.Array,
    denominator: chex.Array,
    eps: chex.Numeric = 1e-10,
) -> chex.Array:
  """Numerically safe division."""
  return numerator / (denominator + eps)


def _placeholder_like(*args):
  return jax.tree_util.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)


def global_clipping(
    clipping_norm: chex.Numeric,
    global_norm_fn: typing.NormFn = optax.global_norm,
    rescale_to_unit_norm: bool = False,
    eps: chex.Numeric = 1e-10,
) -> typing.GradClippingFn:
  """Create a function that clips its input tree to have a maximum L2 norm.

  The L2 norm is computed across leaves of the tree. If the input tree has an L2
  norm that is less or equal to `clipping_norm`, it is left untouched by the
  clipping operation. Otherwise it is scaled down by a positive factor so that
  its new L2 norm is exactly `clipping_norm`.

  Note that the clipping function will return NaN entries if the numerical
  constant `eps` is not small enough. This is to loudly detect loss of
  numerical precision that could lead to invalid results.

  Args:
    clipping_norm: maximum L2 norm to which the input tree should be clipped.
    global_norm_fn: function to compute the L2 norm of an ArrayTree.
    rescale_to_unit_norm: whether the tree should be rescaled to have an L2
      norm of one once it got clipped.
    eps: small numerical constant for numerical stability.
  Returns:
    Function that clips its input tree to have a maximum L2 norm of
    `clipping_norm`.
  """

  def coeff_fn(tree_norm: chex.Array) -> chex.Array:
    one = jnp.ones((), dtype=tree_norm.dtype)
    if rescale_to_unit_norm:
      # coeff = min(1, clipping_norm / tree_norm) / clipping_norm
      return jnp.minimum(
          safe_div(one, clipping_norm, eps),
          safe_div(one, tree_norm, eps)
      )
    else:
      # coeff = min(1, clipping_norm / tree_norm)
      return jnp.minimum(one, safe_div(clipping_norm, tree_norm, eps))

  def clipping_fn(
      grad: typing.ParamsT,
  ) -> tuple[typing.ParamsT, jax.Array]:
    grad_norm = global_norm_fn(grad)
    # If the value of `eps` is invalid because it is too large compared to
    # `clipping_norm`, propagate NaNs to show that the computation is invalid.
    # Note: this has the side effect of always back-propagating NaNs if we
    # differentiate through this function, but this function is not meant to
    # be differentiated, since it post-processes gradients in order to
    # privatize them.
    coeff = jnp.where(clipping_norm > eps, coeff_fn(grad_norm), jnp.nan)
    return jax.tree_util.tree_map(lambda x: x * coeff, grad), grad_norm

  return clipping_fn


def _value_and_clipped_grad_single_sample(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
) -> typing.ValueAndGradFn:
  """Creates a function that computes a clipped gradient for a single sample.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature:
      `(loss, (network_state, metrics)), grads = grad_fn(
          params, network_state, rng_key, inputs)`,
      where `inputs` has a batch dimension. The network state is assumed to be
      independent of the `inputs`, and the metrics are accumulated according to
      their key (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to the gradient.

  Returns:
    Function that computes the gradient for a single (unbatched) sample  and
    clips it. The metrics per sample being returned do *not* have a batch
    dimension.
  """

  def clipped_grad_fn(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    # Add a batch-size dimension.
    inputs_expanded = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0),
        inputs,
    )

    # Compute the gradient.
    (loss, (network_state, metrics)), grad = grad_fn(
        params, network_state, rng_per_example, inputs_expanded)

    clipped_grad, grad_norm = clipping_fn(grad)

    metrics_per_example = {
        'grad_norm': grad_norm,
        # Remove batch dimension in metrics per example.
        **jax.tree_map(functools.partial(jnp.squeeze, axis=0),
                       metrics.per_example),
    }
    # Log the gradient norm per example.
    metrics = dataclasses.replace(
        metrics,
        per_example=metrics_per_example,
    )

    # Apply the clipping function
    return (loss, (network_state, metrics)), clipped_grad

  return clipped_grad_fn


def value_and_clipped_grad_loop(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
    state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
) -> typing.ValueAndGradFn:
  """Create a function that computes grads clipped per example using a loop.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature:
      `(loss, (network_state, metrics)), grads = grad_fn(
          params, network_state, rng_key, inputs)`,
      where `inputs` has a batch dimension. The network state is assumed to be
      independent of the `inputs`, and the metrics are accumulated according to
      their key (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.
    state_acc_strategies: Prefix tree of network state accumulation strategies.

  Returns:
    Function that clips gradient per-example and average them.
  """
  grad_fn_single_sample = _value_and_clipped_grad_single_sample(
      grad_fn=grad_fn,
      clipping_fn=clipping_fn,
  )

  accumulator = grad_clipping_utils.LoopAccumulator(
      grad_fn, state_acc_strategies)

  def clipped_grad_fn(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]
    rng_per_example = jax.random.split(rng_per_example, num=batch_size)

    if batch_size == 1:
      inputs_0 = jax.tree_util.tree_map(
          functools.partial(jnp.squeeze, axis=0),
          inputs,
      )
      (loss, (network_state, metrics)), clipped_grad = grad_fn_single_sample(
          params, network_state, rng_per_example[0], inputs_0)
      metrics = dataclasses.replace(
          metrics,
          # Add batch dimension for metrics per example.
          per_example=jax.tree_map(
              functools.partial(jnp.expand_dims, axis=0),
              metrics.per_example),
      )
      return (loss, (network_state, metrics)), clipped_grad

    def body(value_and_grad, i):
      inputs_i = jax.tree_util.tree_map(lambda x: x[i], inputs)
      value_and_grad_i = grad_fn_single_sample(
          params, network_state, rng_per_example[i], inputs_i)
      value_and_grad = accumulator.accumulate(
          value_and_grad, value_and_grad_i, batch_size, i)
      return value_and_grad, None

    # We only need to know the shape and dtype for the initialization, so we
    # pass the arguments through `_placeholder_like` to make that clear.
    placeholder_args = _placeholder_like(params, network_state,
                                         rng_per_example[0], inputs)
    value_and_grad = accumulator.initialize(
        network_state, batch_size, *placeholder_args)

    # Actually perform the loop.
    value_and_grad, _ = jax.lax.scan(
        body, value_and_grad, jnp.arange(batch_size))
    return value_and_grad

  return clipped_grad_fn


def value_and_clipped_grad_vectorized(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
    state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
) -> typing.ValueAndGradFn:
  """Create a function that computes grads clipped per example using vmapping.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature:
      `(loss, (network_state, metrics)), grads = grad_fn(
          params, network_state, rng_key, inputs)`,
      where `inputs` has a batch dimension. The network state is assumed to be
      independent of the `inputs`, and the metrics are accumulated according to
      their key (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.
    state_acc_strategies: Prefix tree of network state accumulation strategies.

  Returns:
    Function that clips gradient per-example and average them.
  """
  grad_fn_single_sample = _value_and_clipped_grad_single_sample(
      grad_fn=grad_fn,
      clipping_fn=clipping_fn,
  )

  grad_fn_vectorized = jax.vmap(
      grad_fn_single_sample,
      in_axes=(None, None, 0, 0),
  )  # broadcast (params, network_state); vectorise (rng_per_example, inputs)

  def clipped_grad_fn(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    # Compute vectorized outputs and clipped gradients.
    batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]
    rng_per_example = jax.random.split(rng_per_example, num=batch_size)
    value_and_grad = grad_fn_vectorized(
        params, network_state, rng_per_example, inputs)

    return grad_clipping_utils.reduce_vmap(
        state_acc_strategies, value_and_grad, network_state)

  return clipped_grad_fn
