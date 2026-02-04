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

"""Utilities for clipping function outputs and aggregating across a batch."""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import numbers
from typing import Any, Callable, TypeAlias

import chex
import dp_accounting
import jax
import jax.numpy as jnp
import optax


PyTree: TypeAlias = chex.ArrayTree
AuxiliaryOutput = collections.namedtuple('Aux', ['values', 'grad_norms', 'aux'])
_REPLACE_SPECIAL = dp_accounting.NeighboringRelation.REPLACE_SPECIAL


@dataclasses.dataclass(frozen=True)
class BoundedSensitivityCallable:
  """Callable with a sensitivity property.

  If has_aux is False, the sensitivity guarantee holds for the entire output
  which may be an arbitrary pyree of JAX Arrays.  If has_aux is False, the
  output of the function is a pair `(value, aux)` and the sensitivity guarantee
  only holds for `value` PyTree. The aux PyTree is returned on a per-example
  basis (i.e., as a PyTree of arrays having a batch axis).  The caller should
  handle the aux output with care w.r.t. DP guarantees, should they be needed.
  """

  fun: Callable[..., Any]
  l2_norm_bound: float
  has_aux: bool

  def __call__(self, *args, **kwargs):
    return self.fun(*args, **kwargs)

  def sensitivity(
      self,
      neighboring_relation: dp_accounting.NeighboringRelation = _REPLACE_SPECIAL,  # pylint: disable=line-too-long
  ):
    """Returns the L2 sensitivity of the Callable.

    The L2 sensitivity is defined with respect to the given neighboring relation
    and the unit of privacy implied by the function that created this instance.

    Args:
      neighboring_relation: The neighboring relation to consider.

    Returns:
      The L2 sensitivity of the Callable.
    """
    match neighboring_relation:
      case dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE:
        return self.l2_norm_bound
      case dp_accounting.NeighboringRelation.REPLACE_ONE:
        return 2 * self.l2_norm_bound
      case dp_accounting.NeighboringRelation.REPLACE_SPECIAL:
        return self.l2_norm_bound
      case _:
        raise ValueError(f'Unsupported {neighboring_relation=}')


def clip_pytree(
    pytree: PyTree,
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
    nan_safe: bool = True,
    return_zero: bool = False,
):
  """Clips a PyTree of jax arrays based on its global L2 norm.

  Calculates the global L2 norm of the input PyTree. If the norm exceeds
  `clip_norm`, the PyTree is scaled down to have norm equal to `clip_norm`.
  If `rescale_to_unit_norm` is True, the PyTree is additionally scaled by
  `1.0 / clip_norm` (resulting in a norm of at most 1.0 no matter
  what clip_norm is). Handles cases where the original norm is zero,
  or the clip norm is 0 or infinity.

  Formal Guarantees:

  - The output PyTree will have norm at most `clip_norm` if
    `rescale_to_unit_norm` is False, and norm at most 1.0 if it is True.
  - The output PyTree will have the same structure+dtypes as the input PyTree.

  Edge Case Handling:

  ======================= ==================== =================================
  Case                    rescale_to_unit_norm Output
  ======================= ==================== =================================
  clip_norm = 0           False                Zero
  clip_norm = 0           True                 Input / norm, as clip_norm -> 0
  clip_norm = inf         False                Unchanged
  clip_norm = inf         True                 Zero
  clip_norm < 0 (static)  -                    Raises ValueError
  clip_norm < 0 (dynamic) -                    Zero
  pytree_norm = 0         -                    Unchanged
  ======================= ==================== =================================

  Args:
    pytree: The PyTree of arrays to clip.
    clip_norm: The maximum L2 norm allowed.
    rescale_to_unit_norm: If True, the output PyTree's norm is rescaled by `1.0
      / clip_norm` after potential clipping. If False, the output PyTree has
      norm at most `clip_norm`.
    nan_safe: If True, NaNs and +/- infs are converted to 0 before clipping.
      Must be True to preserve the formal guarantees in the presence of NaNs,
      although it does require potentially additional computation. If False, the
      NaNs in input PyTree will be preserved in the output PyTree. +/- infs will
      be converted to NaNs as well.
    return_zero: If True, the output PyTree is guaranteed to be zero no matter
      what the inputs are. Does not influence the formal guarantees.

  Returns:
    A tuple `(clipped_pytree, original_l2_norm)`, where `clipped_pytree` is the
    processed PyTree and `original_l2_norm` is the L2 norm of the input PyTree.
  """
  if isinstance(clip_norm, numbers.Real) and clip_norm < 0:
    raise ValueError(f'clip_norm must be non-negative, got {clip_norm=}.')

  if nan_safe:
    nan_to_num = lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    pytree = jax.tree.map(nan_to_num, pytree)
  clip_norm = jnp.maximum(clip_norm, 0.0)
  l2_norm = optax.global_norm(pytree)
  scale = jnp.minimum(1.0, clip_norm / l2_norm)
  if rescale_to_unit_norm:
    scale = jax.lax.select(clip_norm > 0, scale / clip_norm, 1 / l2_norm)
  # If l2_norm is 0 or nan, set scale to 0.0.
  scale = jnp.nan_to_num(scale, nan=0.0, posinf=0.0)
  clipped = jax.tree.map(lambda x: jnp.astype(scale, x.dtype) * x, pytree)
  maybe_zero = lambda x: jax.lax.select(return_zero, jnp.zeros_like(x), x)
  return jax.tree.map(maybe_zero, clipped), l2_norm.astype(jnp.float32)


# pylint: disable=g-bare-generic
def _with_extra_batch_axis(
    fun: Callable, batch_argnums: int | Sequence[int]
) -> Callable:
  """Wraps a function to add an extra batch axis to the batch_argnums."""
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)

  def wrapped_fun(*args, **kwargs):
    args_with_group_axis = list(args)
    for i in batch_argnums:
      args_with_group_axis[i] = jax.tree.map(
          lambda x: jnp.expand_dims(x, axis=1), args[i]
      )
    return fun(*args_with_group_axis, **kwargs)

  return wrapped_fun


def _validate_batch_args(batch_argnums, args):
  """Validates the arguments to the per-example gradient clipping function."""
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  max_argnum = max(batch_argnums)
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
        'Batch axis must have the same size for all inputs in batch_argnums, '
        f'got {batch_axis_sizes}.'
    )


def _normalize_fun_to_return_aux(fun, has_aux):
  if has_aux:
    return fun
  else:
    return lambda *args, **kwargs: (fun(*args, **kwargs), ())


def _num_real_microbatches(
    is_padding_example: jax.Array,
    microbatch_size: int | None,
) -> int | jax.Array:
  """Calculates the number of non-padding microbatches.

  The returned  result is 1 + the index of the last microbatch that contains at
  least one non-padding example.  This means that microbatches consisting of
  all-padding examples that do not appear at the end will be treated as a real
  microbatch.

  Args:
    is_padding_example: A 1D array of shape (num_examples,).
    microbatch_size: Argument passed to `microbatch`.

  Returns:
    The `true` batch size, as a scalar jax array.
  """
  if microbatch_size is None:
    return is_padding_example.shape[0]
  reshaped = optax.microbatching.reshape_batch_axis(
      is_padding_example, microbatch_size
  )
  # Ensure there is at least one True in the array.
  is_real_batch = jnp.append(True, ~reshaped.all(axis=1))
  # We want the last real microbatch, argmax returns the first True value,
  # so we add increasing numbers from 0 to 1 to each index.
  return jnp.argmax(is_real_batch + jnp.linspace(0, 1, is_real_batch.size))


def clipped_fun(
    fun: Callable,
    has_aux: bool = False,
    *,
    batch_argnums: int | Sequence[int] = 0,
    keep_batch_dim: bool = True,
    l2_clip_norm: float = 1.0,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    return_norms: bool = False,
    microbatch_size: int | None = None,
    nan_safe: bool = True,
    dtype: jax.typing.DTypeLike | None = None,
    prng_argnum: int | None = None,
    spmd_axis_name: str | None = None,
) -> BoundedSensitivityCallable:
  """Transforms a function to clip its output and sum across a batch.

  Example Usage:
    >>> data = jnp.array([0, 1, 2, 3, 4, 5])
    >>> clipped_mean = clipped_fun(jnp.mean, l2_clip_norm=1.0)
    >>> clipped_mean(data)
    Array(5., dtype=float32)

  Formal Guarantees:
    For the first function output:
      The L2 sensitivity of the returned function with respect to the batch
      arguments (specified by `batch_argnums`) under add/remove or zero-out
      differential privacy definitions is guaranteed to be 1.0 if
      `rescale_to_unit_norm` is True. Otherwise, the sensitivity is
      `l2_clip_norm`. Under replace-one DP, the sensitivity is doubled
      (2.0 or 2 * `l2_clip_norm`).
    Extra auxiliary outputs (aux, norms) are per-example. This function
      guarantees that per-example outputs only depend the data for the same
      example. This allows maximum flexibility for the caller to aggregate
      these as desired (possibly with a DP mean, median, quantile, or histogram
      mechanism).

  Args:
    fun: The function to be clipped.
    has_aux: If True, `fun` is expected to return a tuple `(value, aux)`. Only
      the value will be clipped + aggregated, `aux` will be returned on a
      per-example basis. Exercise caution when using this as the sensitivity
      guarantees of the returned Callable are only provided w.r.t. `value`.
    batch_argnums: Specifies which argument(s) of `fun` contain the batch
      dimension. All arguments specified here must have the same size along the
      0th axis.
    keep_batch_dim: If True, batch inputs will be passed to `fun` with a leading
      batch axis of size 1.  If False, this size 1 axis will be dropped
      (reducing the rank of the batch args by 1 before passing to `fun`).
    l2_clip_norm: The maximum L2 norm allowed.
    rescale_to_unit_norm: If True, the output PyTree's norm is rescaled by `1.0
      / clip_norm` after potential clipping. If False, the output PyTree has
      norm at most `clip_norm`.
    normalize_by: Divide the clipped output by this value before returning.
    return_norms: If True, the returned Callable will return the l2_norms of the
      per-example values before clipping. These values should be handled with
      care, see the formal guarantees above.
    microbatch_size: If set, the batch is split up into microbatches of this
      size. These microbatches are then processed sequentially, with operations
      on the groups within each microbatch being vectorized using `vmap`. This
      can be used to reduce peak memory usage at the cost of increased
      sequential computation.
    nan_safe: If True, the formal guarantees of the returned Callable still
      holds in the presence of NaNs and infs. See `clip_pytree` for more details
      on this argument.
    dtype: Optional dtype for the clipped+aggregated pytree. If None, the dtype
      will be the same as the dtypes of the function output. Can be useful to
      avoid overflow issues when using low-precision dtypes as the transformed
      function computes a sum over a potentially large batch.
    prng_argnum: If set, specifies which argument of `fun` is a prng key. The
      prng will be split to have a batch dimension and vmapped over.
    spmd_axis_name: See jax.vmap.

  Returns:
    A new function `clip_fn` that clips the output of `fun` and sums across
    the batch. `clip_fn` takes the same arguments as `fun`. The exact output
    signature depends on `has_aux` and `return_norms`:

    | `has_aux` | `return_norms` | `clipped_fn` returns  |
    | :-------- | :--------------| :-------------------- |
    | `False`   | `False`        | `value`               |
    | `True`    | `False`        | `value, aux`          |
    | `False`   | `True`         | `value, norms`        |
    | `True`    | `True`         | `value, (aux, norms)` |
  """
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)

  fun = _normalize_fun_to_return_aux(fun, has_aux)

  def clipped_fn(*args, **kwargs):
    _validate_batch_args(batch_argnums, args)
    is_padding_example = kwargs.get('is_padding_example', None)
    batch_size = jax.tree.leaves(args[batch_argnums[0]])[0].shape[0]
    if is_padding_example is None:
      is_padding_example = jnp.zeros(batch_size, dtype=jnp.bool_)
      kwargs['is_padding_example'] = is_padding_example

    def clipped_fun_one_group(*args, is_padding_example, **kwargs):
      value, aux = fun(*args, **kwargs)
      value = optax.tree.cast(value, dtype)
      clipped_value, l2_norm = clip_pytree(
          value,
          clip_norm=l2_clip_norm,
          rescale_to_unit_norm=rescale_to_unit_norm,
          nan_safe=nan_safe,
          # See https://arxiv.org/pdf/2411.04205 for info on why this is useful.
          return_zero=is_padding_example,
      )
      return clipped_value, aux, l2_norm

    num_real_mb = _num_real_microbatches(is_padding_example, microbatch_size)
    sum_ = optax.microbatching.AccumulationType.SUM
    concat = optax.microbatching.AccumulationType.CONCAT
    axes = [0 if i in batch_argnums else None for i in range(len(args))]
    if prng_argnum is not None:
      args = list(args)
      rngs = args[prng_argnum]
      split_rngs = jax.tree.map(lambda x: jax.random.split(x, batch_size), rngs)
      args[prng_argnum] = split_rngs
      axes[prng_argnum] = 0

    microbatched_vmap_fun = optax.microbatching.micro_vmap(
        clipped_fun_one_group,
        in_axes=axes,
        microbatch_size=microbatch_size,
        accumulator=(sum_, concat, concat),
        num_real_microbatches=num_real_mb,
        vmap_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
    )

    clipped_values, aux, norms = microbatched_vmap_fun(*args, **kwargs)
    if normalize_by != 1.0:
      clipped_values = jax.tree.map(lambda x: x / normalize_by, clipped_values)

    match has_aux, return_norms:
      case False, False:
        return clipped_values
      case False, True:
        return clipped_values, norms
      case True, False:
        return clipped_values, aux
      case True, True:
        return clipped_values, (aux, norms)

  norm_bound = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by
  if keep_batch_dim:
    clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)
  has_aux = has_aux or return_norms
  return BoundedSensitivityCallable(clipped_fn, norm_bound, has_aux)


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
    prng_argnum: int | None = None,
    spmd_axis_name: str | None = None,
) -> BoundedSensitivityCallable:
  """Create a function to compute the sum of clipped gradients of fun.

  This function acts as a transformation similar to `jax.grad`, but with added
  functionality for gradient clipping applied on a per-example (or per-group)
  basis before summation. It computes the gradient of `fun` with respect to
  `argnums`, calculates the L2 norm of the gradient for each example slice
  along the first axis of the `batch_argnums` args, clips each per-example
  gradient to have a norm of at most `l2_clip_norm`, and finally sums these
  clipped gradients.

  Non-grad outputs of the returned function (aux, values, grad_norms) may
  optionally be returned by setting the arguments `has_aux`,
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

  Example Usage (with Auxiliary Output):
    >>> g = clipped_grad(
    ...   f, l2_clip_norm=jnp.inf, return_values=True, return_grad_norms=True
    ... )
    >>> _, aux = g(3.0, jnp.array([0, 7, -2]))
    >>> aux.values
    Array([ 4.5,  8. , 12.5], dtype=float32)
    >>> aux.grad_norms
    Array([3., 4., 5.], dtype=float32)

  Example Usage (with Per-User Clipping):
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
    rescale_to_unit_norm: If True, clipped gradients are rescaled by `1.0 /
      l2_clip_norm`. This ensures the sensitivity is 1.0. If False, they are
      only scaled down if their norm exceeds `l2_clip_norm`, resulting in a
      sensitivity of `l2_clip_norm`. The motivation for setting this to True is
      to decouple the clipping norm from the learning rate for non-adaptive
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
    prng_argnum: If set, specifies which argumnet of `fun` is a prng key. The
      prng will be split to have a batch dimension and vmapped over.
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
  fun = _normalize_fun_to_return_aux(fun, has_aux)
  value_and_grad_fn = jax.value_and_grad(fun, argnums, has_aux=True)

  def grad_fn(*args, **kwargs):
    value_and_aux, grad = value_and_grad_fn(*args, **kwargs)
    result = pre_clipping_transform(grad)
    if has_aux or return_values or return_grad_norms:
      aux = AuxiliaryOutput(
          values=value_and_aux[0] if return_values else None,
          grad_norms=optax.global_norm(grad) if return_grad_norms else None,
          aux=value_and_aux[1] if has_aux else None,
      )
      return result, aux
    return result

  return clipped_fun(
      grad_fn,
      has_aux=has_aux or return_values or return_grad_norms,
      batch_argnums=batch_argnums,
      l2_clip_norm=l2_clip_norm,
      keep_batch_dim=keep_batch_dim,
      rescale_to_unit_norm=rescale_to_unit_norm,
      normalize_by=normalize_by,
      microbatch_size=microbatch_size,
      nan_safe=nan_safe,
      dtype=dtype,
      prng_argnum=prng_argnum,
      spmd_axis_name=spmd_axis_name,
  )
