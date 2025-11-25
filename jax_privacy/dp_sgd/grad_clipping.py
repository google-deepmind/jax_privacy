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

"""Computing gradients that are clipped per sample."""

import dataclasses
import functools
from typing import Callable, Protocol

from absl import logging
import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping_utils
from jax_privacy.dp_sgd import typing
import optax


class PerExampleGradMethod(Protocol):
  """Version of per-example gradient clipping.

  These methods all give the same results, but have different speed/memory
  trade-offs.
  """

  def __call__(
      self,
      grad_fn: typing.ValueAndGradFn,
      clipping_fn: typing.GradClippingFn,
      state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
  ) -> typing.ValueAndGradFn:
    """Returns a wrapped `grad_fn` that computes gradients clipped per example.

    Args:
      grad_fn: Function that produces unclipped gradients. It is expected to
        have the following signature: `(loss, (network_state, metrics)), grads =
        grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
        batch dimension. The network state is assumed to be independent of the
        `inputs`, and the metrics are accumulated according to their key
        (`per_sample` / `scalars_avg` / `scalars_sum`).
      clipping_fn: clipping function to apply to every per-example gradient
        before those get averaged.
      state_acc_strategies: Prefix tree of network state accumulation
        strategies.

    Returns:
      Function that clips gradient per-example and average them.
    """


def safe_div(
    numerator: chex.Array,
    denominator: chex.Array,
    eps: chex.Numeric = 1e-10,
) -> chex.Array:
  """Numerically safe division."""
  return numerator / (denominator + eps)


def _placeholder_like(*args):
  return jax.tree_util.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args
  )


_ValueAndGradOutputs = tuple[
    tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
    typing.ParamsT,
]


def _map_per_example_metrics(
    f: Callable[[chex.Array], chex.Array], value_and_grad: _ValueAndGradOutputs
) -> _ValueAndGradOutputs:
  """Returns `outputs` with per-example metrics mapped."""
  (loss, (network_state, metrics)), grad = value_and_grad

  metrics_per_example = jax.tree_util.tree_map(f, metrics.per_example)
  metrics = dataclasses.replace(metrics, per_example=metrics_per_example)

  return (loss, (network_state, metrics)), grad


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
  its new L2 norm is exactly `clipping_norm`. Finally, if
  `rescale_to_unit_norm=True`, an additional scaling of `1/cliping_norm` is
  applied so the L2 norm of the returned tree is at most 1.
  Note that the clipping function will return NaN entries if the numerical
  constant `eps` is not small enough. This is to loudly detect loss of
  numerical precision that could lead to invalid results.

  Args:
    clipping_norm: maximum L2 norm to which the input tree should be clipped.
    global_norm_fn: function to compute the L2 norm of an ArrayTree.
    rescale_to_unit_norm: If true, additionally rescale the clipped gradient by
      1/clipping_norm so it has an L2 norm of at most one.
    eps: small numerical constant for numerical stability.

  Returns:
    A `GradClippingFn` that performs the specified clipping operation.
  """

  def compute_scale(tree_norm: chex.Array) -> chex.Array:
    one = jnp.ones((), dtype=tree_norm.dtype)
    if rescale_to_unit_norm:
      # scale = min(1, clipping_norm / tree_norm) / clipping_norm
      return jnp.minimum(
          safe_div(one, clipping_norm, eps), safe_div(one, tree_norm, eps)
      )
    else:
      # scale = min(1, clipping_norm / tree_norm)
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
    scale = jnp.where(clipping_norm > eps, compute_scale(grad_norm), jnp.nan)
    return jax.tree_util.tree_map(lambda x: scale * x, grad), grad_norm

  return clipping_fn


def _batch_size_one_clipped_grad_fn(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
) -> typing.ValueAndGradFn:
  """Creates a function that computes clipped gradients for a batch of size 1.

  If actually invoked with a batch size >1, then this will still proceed and
  compute the clipped gradients for the whole batch. As DP requires per-example
  clipping, though, this is only useful if masking is in place to make the loss
  depend on one input example only.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature: `(loss, (network_state, metrics)), grads =
      grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
      batch dimension. The network state is assumed to be independent of the
      `inputs`, and the metrics are accumulated according to their key
      (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to the gradient.

  Returns:
    Function that computes the gradient for a single sample in a batch of size
    one, and clips it. The metrics per sample being returned *do* have a batch
    dimension.
  """

  def clipped_grad(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    batch_size, *unused_rest = jax.tree_util.tree_leaves(inputs)[0].shape

    # Compute the clipped gradient.
    (loss, (network_state, metrics)), grad = grad_fn(
        params, network_state, rng_per_example, inputs
    )
    clipped_grad, grad_norm = clipping_fn(grad)

    # Log the gradient norm per example.
    metrics_per_example = {
        'grad_norm': jnp.array([grad_norm for _ in range(batch_size)]),
        **metrics.per_example,
    }
    metrics = dataclasses.replace(metrics, per_example=metrics_per_example)

    return (loss, (network_state, metrics)), clipped_grad

  return clipped_grad


def _unbatched_grad_fn(
    grad_fn: typing.ValueAndGradFn,
) -> typing.ValueAndGradFn:
  """Creates a function that computes a gradient for a single unbatched sample.

  Args:
    grad_fn: Function that produces batched gradients. It is expected to have
      the following signature: `(loss, (network_state, metrics)), grads =
      grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
      batch dimension. The network state is assumed to be independent of the
      `inputs`, and the metrics are accumulated according to their key
      (`per_sample` / 'scalars_avg` / `scalars_sum`).

  Returns:
    Function that computes the gradient for a single (unbatched) sample.
    The metrics per sample being returned do *not* have a batch dimension.
  """

  def unbatched_grad(
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

    value_and_grad = grad_fn(
        params, network_state, rng_per_example, inputs_expanded
    )

    # Remove batch dimension in metrics per example.
    return _map_per_example_metrics(
        functools.partial(jnp.squeeze, axis=0),
        value_and_grad,
    )

  return unbatched_grad


def _value_and_clipped_grad_loop(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
    state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
    *,
    select_example_fn: Callable[[int, chex.Array], chex.Array],
    select_example_metric_fn: Callable[[int, chex.Array], chex.Array],
) -> typing.ValueAndGradFn:
  """Create a function that computes grads clipped per example using a loop.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature: `(loss, (network_state, metrics)), grads =
      grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
      batch dimension. The network state is assumed to be independent of the
      `inputs`, and the metrics are accumulated according to their key
      (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.
    state_acc_strategies: Prefix tree of network state accumulation strategies.
    select_example_fn: Maps tensor to tensor of the same rank, slicing or
      masking along the batch dimension to build a new batch comprising a single
      example.
    select_example_metric_fn: Maps tensor to tensor of rank one lower, selecting
      or squeezing along the batch dimension to extract the value for the single
      example.

  Returns:
    Function that clips gradient per-example and average them.
  """
  batch_size_one_clipped_grad_fn = _batch_size_one_clipped_grad_fn(
      grad_fn=grad_fn,
      clipping_fn=clipping_fn,
  )

  accumulator = grad_clipping_utils.LoopAccumulator(
      grad_fn, state_acc_strategies
  )

  def clipped_grad_fn(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    batch_size, *unused_rest = jax.tree_util.tree_leaves(inputs)[0].shape
    rng_per_example = jax.random.split(rng_per_example, num=batch_size)

    if batch_size == 1:
      return batch_size_one_clipped_grad_fn(
          params, network_state, rng_per_example[0], inputs
      )

    def body(value_and_grad, i):
      # Select the current example.
      inputs_i = jax.tree_util.tree_map(
          functools.partial(select_example_fn, i), inputs
      )

      # Compute clipped gradients of the single-example batch.
      value_and_grad_i = batch_size_one_clipped_grad_fn(
          params, network_state, rng_per_example[i], inputs_i
      )

      # Select the current example's metrics.
      value_and_grad_i = _map_per_example_metrics(
          functools.partial(select_example_metric_fn, i), value_and_grad_i
      )

      return (
          accumulator.accumulate(
              value_and_grad, value_and_grad_i, batch_size, i
          ),
          None,
      )

    # We only need to know the shape and dtype for the initialization, so we
    # pass the arguments through `_placeholder_like` to make that clear.
    placeholder_args = _placeholder_like(
        params, network_state, rng_per_example[0], inputs
    )
    value_and_grad = accumulator.initialize(
        network_state, batch_size, *placeholder_args
    )

    # Actually perform the loop.
    value_and_grad, _ = jax.lax.scan(
        body, value_and_grad, jnp.arange(batch_size)
    )
    return value_and_grad

  return clipped_grad_fn


def _value_and_clipped_grad_vmap_of_reshape(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
    state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
    *,
    spmd_axis_name: str | tuple[str, ...] | None = None,
) -> typing.ValueAndGradFn:
  """Create a function that computes grads clipped per example using vmapping.

  This uses a pattern where we vmap over a function that reshapes the batch into
  per-example singleton batches.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature: `(loss, (network_state, metrics)), grads =
      grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
      batch dimension. The network state is assumed to be independent of the
      `inputs`, and the metrics are accumulated according to their key
      (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.
    state_acc_strategies: Prefix tree of network state accumulation strategies.
    spmd_axis_name: Optional vectorisation axis to pass to `jax.vmap`.

  Returns:
    Function that clips gradient per-example and average them.
  """
  grad_fn_single_sample = _unbatched_grad_fn(
      _batch_size_one_clipped_grad_fn(
          grad_fn=grad_fn,
          clipping_fn=clipping_fn,
      ),
  )

  grad_fn_vectorized = jax.vmap(
      grad_fn_single_sample,
      in_axes=(None, None, 0, 0),
      spmd_axis_name=spmd_axis_name,
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
        params, network_state, rng_per_example, inputs
    )

    return grad_clipping_utils.reduce_vmap(
        state_acc_strategies, value_and_grad, network_state
    )

  return clipped_grad_fn


def _value_and_clipped_grad_reshape_then_vmap(
    grad_fn: typing.ValueAndGradFn,
    clipping_fn: typing.GradClippingFn,
    state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
    *,
    spmd_axis_name: str | tuple[str, ...] | None = None,
    use_shard_alike: bool = False,
) -> typing.ValueAndGradFn:
  """Create a function that computes grads clipped per example using vmap.

  This uses a pattern where we first reshape the batch into per-example
  singleton batches, then vmap over those.

  Args:
    grad_fn: Function that produces unclipped gradients. It is expected to have
      the following signature: `(loss, (network_state, metrics)), grads =
      grad_fn( params, network_state, rng_key, inputs)`, where `inputs` has a
      batch dimension. The network state is assumed to be independent of the
      `inputs`, and the metrics are accumulated according to their key
      (`per_sample` / 'scalars_avg` / `scalars_sum`).
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.
    state_acc_strategies: Prefix tree of network state accumulation strategies.
    spmd_axis_name: Optional vectorisation axis to pass to `jax.vmap`.
    use_shard_alike: Whether to use `shard_alike` to ensure vmap outputs are
      sharded the same way as its inputs.

  Returns:
    Function that clips gradient per-example and average them.
  """
  single_element_batch_clipped_grad_fn = _batch_size_one_clipped_grad_fn(
      grad_fn=grad_fn, clipping_fn=clipping_fn
  )

  def grad_fn_single_batch(
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng_per_example: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[
      tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
      typing.ParamsT,
  ]:
    value_and_grad = single_element_batch_clipped_grad_fn(
        params, network_state, rng_per_example, inputs
    )
    # Remove batch dimension in metrics per example.
    return _map_per_example_metrics(
        functools.partial(jnp.squeeze, axis=0),
        value_and_grad,
    )

  if use_shard_alike:
    try:  # We don't want a hard dependency on drjax, so import it here.
      import drjax  # pylint: disable=g-import-not-at-top, import-outside-toplevel
    except ImportError:
      raise ImportError(
          'Could not import drjax. Please install it if you want to use the'
          ' `sharded_vectorized_grad_method` method with use_shard_alike=True. '
          ' You can install it with `pip install drjax`.'
      ) from None
    if spmd_axis_name is None:
      raise ValueError(
          'spmd_axis_name must be provided if use_shard_alike is True.'
      )

    def grad_fn_vectorized(
        params: typing.ParamsT,
        network_state: typing.ModelStateT,
        rng_per_example: chex.PRNGKey,
        inputs: typing.InputsT,
    ) -> tuple[
        tuple[typing.Loss, tuple[typing.ModelStateT, typing.Metrics]],
        typing.ParamsT,
    ]:
      grad_fn_with_params_and_state = functools.partial(
          grad_fn_single_batch, params, network_state
      )
      batch_size = jax.tree.leaves(inputs)[0].shape[0]
      logging.info('Inferred batch size for drjax.map_fn: %d', batch_size)

      @drjax.program(
          placements={spmd_axis_name: batch_size},
      )
      def map_grad_fn(x, y):
        return drjax.map_fn(grad_fn_with_params_and_state, (x, y))

      return map_grad_fn(rng_per_example, inputs)

  else:

    grad_fn_vectorized = jax.vmap(
        grad_fn_single_batch,
        in_axes=(None, None, 0, 0),
        spmd_axis_name=spmd_axis_name,
    )

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
    inputs_with_extra_axis = jax.tree.map(
        lambda x: jnp.expand_dims(x, axis=1), inputs
    )
    rng_per_example = jax.random.split(rng_per_example, num=batch_size)
    value_and_grad = grad_fn_vectorized(
        params, network_state, rng_per_example, inputs_with_extra_axis
    )

    return grad_clipping_utils.reduce_vmap(
        state_acc_strategies, value_and_grad, network_state
    )

  return clipped_grad_fn


def unrolled_grad_method(
    *,
    select_example_fn: Callable[[int, chex.Array], chex.Array],
    select_example_metric_fn: Callable[[int, chex.Array], chex.Array],
) -> PerExampleGradMethod:
  return functools.partial(
      _value_and_clipped_grad_loop,
      select_example_fn=select_example_fn,
      select_example_metric_fn=select_example_metric_fn,
  )


def vectorized_grad_method(
    *,
    spmd_axis_name: str | tuple[str, ...] | None = None,
) -> PerExampleGradMethod:
  return functools.partial(
      _value_and_clipped_grad_vmap_of_reshape,
      spmd_axis_name=spmd_axis_name,
  )


VECTORIZED: PerExampleGradMethod = vectorized_grad_method()
"""Use `vmap`. Faster, but needs more memory."""


def sharded_vectorized_grad_method(
    *,
    spmd_axis_name: str | tuple[str, ...],
) -> PerExampleGradMethod:
  """A vmap-based per-example grad method with intermediate sharding info.

  This can be significantly faster than the default vectorized method,
  especially when the underlying grad_fn is sharded, but requires the user to
  provide an `spmd_axis_name` to ensure that the intermediate sharding
  annotations are correct. For more details, see `spmd_axis_name` in `jax.vmap`.

  Args:
    spmd_axis_name: The mesh axis over which to vectorize.

  Returns:
    A `PerExampleGradMethod`.
  """
  return functools.partial(
      _value_and_clipped_grad_reshape_then_vmap,
      spmd_axis_name=spmd_axis_name,
      use_shard_alike=True,
  )


UNROLLED: PerExampleGradMethod = unrolled_grad_method(
    select_example_fn=lambda i, x: jnp.expand_dims(x[i], axis=0),
    select_example_metric_fn=lambda i, x: jnp.squeeze(x, axis=0),
)
"""Use `lax.scan` to unroll over examples. Uses less memory."""
