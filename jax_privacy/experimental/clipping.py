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

from collections.abc import Sequence
import dataclasses
import functools
import numbers
from typing import Any, Callable, TypeAlias

import chex
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy.experimental import microbatching
import optax


PyTree: TypeAlias = chex.ArrayTree
_REPLACE_SPECIAL = dp_accounting.NeighboringRelation.REPLACE_SPECIAL


@dataclasses.dataclass(frozen=True)
class BoundedSensitivityCallable:
  """Callable with a sensitivity property.

  The function may return multiple outputs, some of which may have a batch
  axis and some of which may not. The sensitivity guarantee holds for all
  outputs that do not have a batch axis (i.e., because there was an aggregation
  over it.)  The auxiliary outputs with a batch axis are usually computed
  essentially for free so they are returned here, but must be handled with care
  by the caller (with respect to the DP guarantees, should they be needed).
  """

  fun: Callable[..., Any]
  l2_norm_bound: float

  def __call__(self, *args, **kwargs):
    return self.fun(*args, **kwargs)

  def sensitivity(
      self,
      neighboring_relation: dp_accounting.NeighboringRelation = _REPLACE_SPECIAL
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

  Edge Case Handling:
  - clip_norm = 0:
    - rescale_to_unit_norm = False: The output PyTree is zero.
    - rescale_to_unit_norm = True: The output PyTree is equal to the input
      PyTree divided by its norm (the limiting behavior as clip_norm -> 0).
  - clip_norm = inf:
    - rescale_to_unit_norm = False: The output PyTree is unchanged.
    - rescale_to_unit_norm = True: The output PyTree is zero.
  - pytree_norm = 0: The output PyTree is unchanged.
  - clip_norm < 0:
    - If clip_norm is a static input (python float), raises a ValueError.
    - If clip_norm is a dynamic input (jax.Array or Tracer), it is treated as 0.

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
  return jax.tree.map(lambda x: scale * x, pytree), l2_norm.astype(jnp.float32)


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


def clip_sum(
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
    spmd_axis_name: str | None = None,
) -> BoundedSensitivityCallable:
  """Transforms a function to clip its output and sum across a batch.

  Example Usage:
    >>> data = jnp.array([0, 1, 2, 3, 4, 5])
    >>> clipped_mean = clip_sum(jnp.mean, l2_clip_norm=1.0)
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
    | `True`    | `True`         | `value, aux, norms`   |
  """
  if isinstance(batch_argnums, int):
    batch_argnums = (batch_argnums,)
  if not has_aux:
    fun_with_aux = lambda *args, **kwargs: (fun(*args, **kwargs), ())
  else:
    fun_with_aux = fun

  def clipped_fn(*args, **kwargs):
    is_padding_example = kwargs.get('is_padding_example', None)
    if is_padding_example is None:
      batch_size = jax.tree.leaves(args[batch_argnums[0]])[0].shape[0]
      kwargs['is_padding_example'] = jnp.zeros(batch_size, dtype=jnp.bool_)

    def clipped_fun_one_group(*args, is_padding_example, **kwargs):
      value, aux = fun_with_aux(*args, **kwargs)
      clipped_value, l2_norm = jax.lax.cond(
          is_padding_example,
          # See https://arxiv.org/pdf/2411.04205 for info on why this is useful.
          lambda pytree: (
              jax.tree.map(jnp.zeros_like, pytree),
              jnp.array(0.0, dtype=jnp.float32),
          ),
          functools.partial(
              clip_pytree,
              clip_norm=l2_clip_norm,
              rescale_to_unit_norm=rescale_to_unit_norm,
              nan_safe=nan_safe,
          ),
          value,
      )
      return clipped_value, aux, l2_norm

    sum_ = microbatching.AccumulationType.SUM
    concat = microbatching.AccumulationType.CONCAT
    axes = [0 if i in batch_argnums else None for i in range(len(args))]
    microbatched_vmap_fun = microbatching.inmemory_microbatched_fn_general(
        jax.vmap(clipped_fun_one_group, axes, spmd_axis_name=spmd_axis_name),
        batch_argnums=batch_argnums,
        microbatch_size=microbatch_size,
        accumulation_type=(sum_, concat, concat),
    )

    clipped_values, aux, norms = microbatched_vmap_fun(*args, **kwargs)
    # It would save flops to call this after vmap but before the microbatching.
    result = jax.tree.map(lambda x: jnp.sum(x, 0, dtype), clipped_values)
    if normalize_by != 1.0:
      result = jax.tree.map(lambda x: x / normalize_by, result)

    match has_aux, return_norms:
      case False, False:
        return result
      case False, True:
        return result, norms
      case True, False:
        return result, aux
      case True, True:
        return result, aux, norms

  norm_bound = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by
  if keep_batch_dim:
    clipped_fn = _with_extra_batch_axis(clipped_fn, batch_argnums)
  return BoundedSensitivityCallable(clipped_fn, norm_bound)
