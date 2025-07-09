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

"""Simple wrapper around optax to be used for strategy optimization."""

from collections.abc import Callable
import dataclasses
from typing import Any, TypeAlias, TypeVar

import chex
import jax
import jax.numpy as jnp
import optax

ParamT = TypeVar('ParamT', bound=chex.ArrayTree)

DEFAULT_OPTIMIZER = optax.lbfgs(
    memory_size=1, linesearch=optax.scale_by_backtracking_linesearch(128)
)


@dataclasses.dataclass
class CallbackArgs:
  """Information passed to the callback function on each optimization step.

  Properties:
    step: The current optimization step.
    loss: The loss value at the current step.
    grad: The gradient at the current step.
    params: The current parameters.
    state: The current optimizer state.
  """

  step: int
  loss: jnp.ndarray
  grad: chex.ArrayTree | None
  params: chex.ArrayTree
  state: Any


CallbackFnType: TypeAlias = Callable[[CallbackArgs], None | bool]


def jax_enable_x64(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Decorator to enable x64 precision for a function."""

  def wrapped_fn(*args, **kwargs):
    with jax.experimental.enable_x64():
      return fn(*args, **kwargs)

  return wrapped_fn


@jax_enable_x64
def optimize(
    loss_fn: Callable[
        [ParamT], jnp.ndarray | tuple[jnp.ndarray, ParamT]
    ],
    params: ParamT,
    *,
    max_optimizer_steps: int = 250,
    grad: bool = False,
    callback: CallbackFnType = lambda _: None,
    optimizer: optax.GradientTransformationExtraArgs = DEFAULT_OPTIMIZER,
) -> ParamT:
  """Optimize a differentiable loss function using L-BFGS.

  This is a simple wrapper around optax.  It automatically enables x64 precision
  and JIT-compiles the objective function, gradient, and update rule.
  The default solver (L-BFGS) works well for the strategy classes supported in
  this codebase, and has generally been observed to work well for
  matrix-factorization-type problems
  in the past.  See e.g.,
    * https://arxiv.org/abs/2106.12118
    * https://arxiv.org/abs/2405.15913
    * https://arxiv.org/abs/2306.08153
    * https://arxiv.org/abs/2408.08868

  Args:
    loss_fn: A loss function to minimize.
    params: Initial parameters.  These will be cast to float64 internally.
    max_optimizer_steps: The (maximum) number of optimization steps.
    grad: Flag indicating if the loss_fn also returns the gradient.
    callback: Optional callback function to call after each optimization step.
      The callback will be called after each iteration with a `CallbackArgs`
      dataclass.  Early stopping can be achieved by having the callback return a
      truthy value.
    optimizer: An optax.GradientTransformation to use as the underlying
      optimizer.

  Returns:
    The parameters that approximately locally minimize the given loss_fun,
    casted back to the same types as the original `params`.
  """
  loss_and_grad = loss_fn if grad else jax.value_and_grad(loss_fn)
  value_fn = (lambda x: loss_fn(x)[0]) if grad else loss_fn

  @jax.jit
  def single_step(params, opt_state):
    value, grad = loss_and_grad(params)
    updates, opt_state = optimizer.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=value_fn
    )
    new_params = optax.apply_updates(params, updates)
    return value, grad, new_params, opt_state

  # MF-style strategy optimization problems are numerically sensitive to
  # precision, so we use f64 internally.
  original_dtypes = jax.tree.map(jnp.dtype, params)

  params = jax.tree.map(jnp.float64, params)
  state = optimizer.init(params)
  for i in range(max_optimizer_steps):
    loss, grad, params, state = single_step(params, state)
    if callback(
        CallbackArgs(step=i, loss=loss, grad=grad, params=params, state=state)
    ):
      break

  return jax.tree.map(jnp.astype, params, original_dtypes)
