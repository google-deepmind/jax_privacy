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

"""Experimental per-example gradient clipping API.

See README for an overview of this module.
"""

import collections
from collections.abc import Sequence
from typing import Callable, TypeAlias

import chex
import jax
from jax_privacy.experimental import clipping


PyTree: TypeAlias = chex.ArrayTree
AuxiliaryOutput = collections.namedtuple('Aux', ['values', 'grad_norms', 'aux'])


def _validate_static_args(argnums, batch_argnums, normalize_by):
  """Validates the argnums and batch_argnums inputs are compatible."""
  if normalize_by <= 0.0:
    raise ValueError(f'normalize_by must be > 0, got {normalize_by}.')
  if isinstance(argnums, int):
    argnums = (argnums,)
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  if not batch_argnums:
    raise ValueError('Batch Argnums must not be empty.')
  if min(argnums + batch_argnums) < 0:
    raise ValueError(
        f'argnums={argnums} and batch_argnums={batch_argnums} must be >= 0.'
    )
  shared_argnums = set(argnums) & set(batch_argnums)
  if shared_argnums:
    raise ValueError(
        'Cannot compute clipped gradients for argnums that have a batch axis. '
        f'{argnums=} and {batch_argnums=} with overlap {list(shared_argnums)}.'
    )


def _validate_args(argnums, batch_argnums, args):
  """Validates the arguments to the per-example gradient clipping function."""
  if isinstance(argnums, int):
    argnums = (argnums,)
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  max_argnum = max(argnums + batch_argnums)
  if len(args) <= max_argnum:
    raise ValueError(
        f'Unable to find argnum={max_argnum}, was given {len(args)} args.'
    )

  batch_args = [args[i] for i in batch_argnums]
  batch_axis_sizes = set(
      jax.tree.flatten(jax.tree.map(lambda x: x.shape[0], batch_args))[0]
  )
  if len(batch_axis_sizes) > 1:
    raise ValueError(
        f'Batch axis must have the same size for all inputs in batch_argnums, '
        f'got {batch_axis_sizes}.'
    )


# pylint: disable=g-bare-generic
def clipped_grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    *,
    l2_clip_norm: float,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    batch_argnums: int | Sequence[int] = 1,
    keep_batch_dim: bool = True,
    return_values: bool = False,
    return_grad_norms: bool = False,
    pre_clipping_transform: Callable[[PyTree], PyTree] = lambda x: x,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: jax.typing.DTypeLike | None = None,
    spmd_axis_name: str | None = None,
) -> clipping.BoundedSensitivityCallable:
  """Create a function to compute the sum of clipped gradients of fun.

  This function acts as a transformation similar to `jax.grad`, but with added
  functionality for gradient clipping applied on a per-example (or per-group)
  basis before summation. It computes the gradient of `fun` with respect to
  `argnums`, calculates the L2 norm of the gradient for each example slice
  along the first axis of the `batch_argnums` args, clips each per-example
  gradient to have a norm of at most `l2_clip_norm`, and finally sums these
  clipped gradients.

  Non-grad outputs of the returned function (aux, values, grad_norms) may
  optionally be returned by setting the argumumnets `has_aux`, 
  `return_values`, and/or `return_grad_norms` to True.  These outputs are
  per-example, and hence have a batch axis. It is up to the caller to handle
  these as necessary. See the `DP Sensitivity Guarantee` below for more details
  on this design choice.

  Example Usage:
    >>> import jax.numpy as jnp 
    >>> f = lambda param, data: 0.5 * jnp.mean((data - param)**2)
    >>> g = clipped_grad(f, l2_clip_norm=jnp.inf)
    >>> g(3.0, jnp.array([0, 7, -2]))
    Array(4., dtype=float32)

  Example Usage (Per-User Clipping):
    >>> f = lambda param, data: 0.5 * jnp.mean((data - param)**2)
    >>> g = clipped_grad(f, l2_clip_norm=jnp.inf, keep_batch_dim=False)
    >>> userA = jnp.array([1, -1])
    >>> userB = jnp.array([2, 2])
    >>> userC = jnp.array([0, 3])
    >>> g(3.0, jnp.array([userA, userB, userC]))
    Array(5.5, dtype=float32)

  Formal Guarantees:
    For the gradient output:
      The L2 sensitivity of the returned function with respect to the batch
      arguments (specified by `batch_argnums`) under add/remove or zero-out
      differential privacy definitions is guaranteed to be 1.0 if
      `rescale_to_unit_norm` is True. Otherwise, the sensitivity is
      `l2_clip_norm`. Under replace-one DP, the sensitivity is doubled
      (2.0 or 2 * `l2_clip_norm`).
    All auxiliary outputs (aux, values, grad_norms) are per-example. This
      function guarantees that per-example outputs only depend the data for the
      same example. This allows maximum flexibility for the caller to aggregate
      these as desired (possibly with a DP mean, median, quantile, or histogram
      mechanism).

  Args:
    fun: The function to be differentiated, which should return a scalar loss
      value. If `has_aux` is True, it should return a tuple `(value, aux)`.
    argnums: Specifies which argument(s) of `fun` to differentiate with respect
      to. Can be an integer or a sequence of integers. These arguments should
      *not* have a batch dimension.
    has_aux: If True, `fun` is expected to return a tuple `(value, aux)`. The
      auxiliary data `aux` will be returned by the transformed function.
      Exercise caution when using this as no DP sensitivity guarantees are
      provided for the auxiliary data.
    l2_clip_norm: The maximum L2 norm for each per-example gradient. Gradients
      with a norm larger than this value will be scaled down.
    rescale_to_unit_norm: If True, clipped gradients are rescaled by
      `1.0 / l2_clip_norm`. This ensures the sensitivity is 1.0. If False, they
      are only scaled down if their norm exceeds `l2_clip_norm`, resulting in a
      sensitivity of `l2_clip_norm`. The motivation for setting this to True
      is to decouple the clipping norm from the learning rate for non-adaptive
      optimizers, as described in https://arxiv.org/abs/2204.13650.
    normalize_by: Divide the clipped output by this value before returning.
    batch_argnums: Specifies which argument(s) of `fun` contain the batch
      dimension (usually the data and labels). Can be an integer or a sequence
      of integers. All arguments specified here must have the same size along
      their first dimension (the batch dimension) the default value of 1 assumes
      the signature of fun is `fun(params, batch)`.
    keep_batch_dim: If True, batch inputs will be passed to `fun` with a leading
      batch axis of size 1.  If False, this size 1 axis will be dropped
      (reducing the rank of the batch args by 1 before passing to `fun`). The
      default value of True assumes that `fun` expects inputs with a batch axis.
      Overriding this default can be useful if fun defines the loss function for
      a single example, or if clipping should be applied at the group or user
      level (in which case an extra batch axis is added to the inputs).
    return_values: If True, the transformed function will also return the
      per-example values, before clipping.
    return_grad_norms: If True, the transformed function will also return the
      per-example gradient norms, before clipping.
    pre_clipping_transform: An optional function to apply to the per-example
      gradients before clipping. The function should consume the gradient pytree
      for a single example and returned a new pytree (possibly with different
      structure). Can be used to e.g., scale the leaves of the pytree to
      accommodate preconditioner clipping. Does not affect the sensitivity
      guarantee.
    microbatch_size: If set, input groups are formed into microbatches of this
      size. These microbatches are then processed sequentially, with operations
      on the groups within each microbatch being vectorized using `vmap`. This
      can be used to reduce peak memory usage at the cost of increased
      sequential computation. Microbatching will be at the level of
      users/groups.  E.g., if there are 500 users, with 7 examples per user, and
      microbatch_size=100, then the input will be broken into 5 microbatches of
      100 users, and when processing a microbatch, `fun` will be invoked 100
      times (in parallel with vmap) on groups of 7 examples.
    nan_safe: If True, the formal guarantees of the returned Callable still
      holds in the presence of NaNs and infs. See `clip_pytree` for more details
      on this argument.
    dtype: Optional dtype for the returned gradient. If None, the dtype will be
      the same as the dtypes of the gradient function. Can be useful to avoid
      overflow issues when using low-precision dtypes as the returned function
      computes a sum over a potentially large batch.
    spmd_axis_name: See jax.vmap. Only relevant in distributed settings.

  Returns:
    A new function `values_and_clipped_grad_fn` that computes the sum of clipped 
    per-group gradients of `fun`. The returned function returns `grad`
    if return_values = return_grad_norms = has_aux = False.  Otherwise, it 
    returns a tuple of grad, AuxiliaryOutput, where AuxiliaryOutput is a
    namedtuple with optional fields (values, grad_norms, aux) containing the
    per-example values, gradient norms, and auxiliary data, respectively.
  """
  _validate_static_args(argnums, batch_argnums, normalize_by)
  value_and_grad_fn = jax.value_and_grad(fun, argnums, has_aux=has_aux)
  def grad_fn(*args, **kwargs):
    value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
    return pre_clipping_transform(grad), value_and_aux

  clipped_grad_fn = clipping.clip_sum(
      grad_fn,
      has_aux=True,
      batch_argnums=batch_argnums,
      l2_clip_norm=l2_clip_norm,
      keep_batch_dim=keep_batch_dim,
      rescale_to_unit_norm=rescale_to_unit_norm,
      normalize_by=normalize_by,
      return_norms=True,
      microbatch_size=microbatch_size,
      nan_safe=nan_safe,
      dtype=dtype,
      spmd_axis_name=spmd_axis_name,
  )

  def wrapped_clipped_grad_fn(*args, **kwargs):
    _validate_args(argnums, batch_argnums, args)
    grad, values_and_maybe_aux, norms = clipped_grad_fn(*args, **kwargs)
    values = values_and_maybe_aux[0] if has_aux else values_and_maybe_aux
    per_example_aux = AuxiliaryOutput(
        values=values if return_values else None,
        grad_norms=norms if return_grad_norms else None,
        aux=values_and_maybe_aux[1] if has_aux else None,
    )
    if has_aux or return_values or return_grad_norms:
      return grad, per_example_aux
    return grad

  return clipping.BoundedSensitivityCallable(
      wrapped_clipped_grad_fn,
      clipped_grad_fn.l2_norm_bound,
  )
