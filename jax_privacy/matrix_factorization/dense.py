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

"""Optimization and error fns for dense (explicitly represented) strategies.

See `sensitivity.py` for sensitivity calculations for dense strategies.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from . import checks
from . import optimization
from . import sensitivity


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def per_query_error(
    *,
    strategy_matrix: jax.Array | None = None,
    noising_matrix: jax.Array | None = None,
    workload_matrix: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Expected per-query squared error for a general matrix mechanism.

  Exactly one of `strategy_matrix` and `noising_matrix` should be provided.

  Args:
    strategy_matrix: The (square) strategy matrix C defining the mechanism.
    noising_matrix: The (possibly non-square) noising matrix C^{-1}.
    workload_matrix: The workload matrix. Defaults to `jnp.tri`, the prefix sum
      workload matrix.
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected per-query squared error, an array of length n.
  """
  if not skip_checks:
    if (strategy_matrix is None) == (noising_matrix is None):
      raise ValueError(
          'Specify exactly one of strategy_matrix or noising_matrix.'
      )

  if strategy_matrix is not None:
    C = strategy_matrix
    if not skip_checks:
      # We require square C, because for a non-square C, the choice of the
      # specific pseudoinverse C^{-1} determines the error. Hence, we require
      # the user to specify the C^{-1} explicitly and pass that in instead.
      checks.check_square(C, 'strategy_matrix')
    n = C.shape[1]
    A = workload_matrix if workload_matrix is not None else jnp.tri(n)
    # Solve B @ C = A for B
    B = jnp.linalg.solve(C.T, A.T).T
    if not skip_checks:
      checks.check(A=A, B=B, C=C)
  else:
    assert noising_matrix is not None
    C_inv = noising_matrix
    n = C_inv.shape[0]
    A = workload_matrix if workload_matrix is not None else jnp.tri(n)
    B = A @ C_inv
    if not skip_checks:
      checks.check(A=A, B=B)

  return jnp.sum(B * B, axis=1)


def max_error(
    *,
    strategy_matrix: jax.Array | None = None,
    noising_matrix: jax.Array | None = None,
    workload_matrix: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Max-over-iterations squared error for a general matrix mechanism.

  Exactly one of `strategy_matrix` and `noising_matrix` should be provided.

  Args:
    strategy_matrix: The (square) strategy matrix C defining the mechanism.
    noising_matrix: The (possibly non-square) noising matrix C^{-1}.
    workload_matrix: The workload matrix. Defaults to `jnp.tri`, the prefix sum
      workload matrix.
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected max-over-iterations squared error (a scalar).
  """
  return per_query_error(
      strategy_matrix=strategy_matrix,
      noising_matrix=noising_matrix,
      workload_matrix=workload_matrix,
      skip_checks=skip_checks,
  ).max()


def mean_error(
    *,
    strategy_matrix: jax.Array | None = None,
    noising_matrix: jax.Array | None = None,
    workload_matrix: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Mean-over-iterations squared error for a general matrix mechanism.

  Exactly one of `strategy_matrix` and `noising_matrix` should be provided.

  Args:
    strategy_matrix: The (square) strategy matrix C defining the mechanism.
    noising_matrix: The (possibly non-square) noising matrix C^{-1}.
    workload_matrix: The workload matrix. Defaults to `jnp.tri`, the prefix sum
      workload matrix.
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected mean-over-iterations squared error (a scalar).
  """
  return per_query_error(
      strategy_matrix=strategy_matrix,
      noising_matrix=noising_matrix,
      workload_matrix=workload_matrix,
      skip_checks=skip_checks,
  ).mean()


def get_orthogonal_mask(n: int, epochs: int = 1) -> jax.Array:
  """Computes a mask that imposes orthognality constraints on the optimization.

  This is specific to the fixed-epoch-order (k, b)-participation schema of
  https://arxiv.org/pdf/2211.06530.pdf, where participations are separated by
  exactly b-1 steps, and b = n / epochs.

  This mask sets entry M_{ij} = 0 if i == j (mod b) and M_{ij} = 1
  otherwise.  Sensitivity for any matrix with 0s in these entries is easy to
  calculate as only a function of the diagonal.  Moreover, the sensitivity is
  equal for all possible {-1,1} participation vectors.

  Args:
    n: the size of the mask
    epochs: The number of epochs

  Returns:
    A 0/1 mask
  """
  # We use numpy instead of Jax internally here because we are performing
  # in-place updates to mask.
  mask = np.ones((n, n))
  b = n // epochs
  for i in range(b):
    mask[i::b, i::b] = np.eye(epochs)
  return jnp.array(mask)


def _mean_loss_and_gradient(
    X: jax.Array, A: jax.Array
) -> tuple[jax.Array, jax.Array]:
  r"""Computes the matrix mechanism total squared error loss and gradient.

  This function computes $\tr[A^T A X^{-1}]$ and the associated gradient
  $dX = -X^{-1} A^T A X^{-1}$.  It assumes that $X$ is a symmetric positive
  definite matrix.  For efficiency, no error is thrown if this assumption is
  not satisfied, but the returned loss or gradient may contain NaN's if this
  is the case.

  Args:
    X: The current iterate, an n x n symmetric positive definite matrix.
    A: The workload, an n x n matrix

  Returns:
    loss: a real-valued number
    gradient: the gradient of the loss w.r.t. X, an n x n matrix
  """
  # It is significantly faster to compute the gradient ourselves rather than
  # rely on Jax autodiff here. For n=8192, difference is 550ms vs. 900ms on GPU.
  n = X.shape[0]
  H = jsp.linalg.solve(X, A.T, assume_a='pos')
  return jnp.trace(H @ A) / n, -H @ H.T / n


def strategy_from_X(X: jax.Array) -> jax.Array:
  """Return a lower triangular strategy matrix C from its Gram matrix.

  Args:
    X: A positive symmetric semidefinite matrix.

  Returns:
    A lower triangular matrix C satisfying X = C^T C.
  """
  return jnp.linalg.cholesky(X[::-1, ::-1]).T[::-1, ::-1]


def pg_tol_termination_fn(step_info: optimization.CallbackArgs) -> bool:
  """Callback function that returns True if projected gradient is near-zero."""
  return bool(jnp.abs(step_info.grad).max() <= 1e-3)


def optimize(
    n: int,
    *,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
    A: jax.Array | None = None,
    max_optimizer_steps: int = 10000,
    callback: optimization.CallbackFnType = pg_tol_termination_fn,
) -> jax.Array:
  """Optimizes a strategy matrix C for mean loss and a participation pattern.

  Currently only MSE (mean error) is supported.

  This function can be used to optimize matrices under
  * Single-participation:
  [Denisov et al., 2022](https://arxiv.org/abs/2202.08312).  This can be
  accomplished by running with default arguments.

  * Multi-participation with fixed-epoch order
  [Choquette-Choo et al., 2022](https://arxiv.org/abs/2211.06530).
  This can be accomplished by setting epochs=k.

  * Multi-participation with min-separation (useful for federated training
  scenarios).  This can be accomplished by setting bands = min_sep and
  equal_norm = True.

  * Multi-participation with amplification via subsampled fixed-epoch order
  [Choquette-Choo et al., 2022](https://arxiv.org/abs/2211.06530).  This can be
  accomplished by setting epochs = 1, bands < separation, and equal_norm = True.

  Args:
    n: the number of iterations the strategy should encode.
    epochs: The number of epochs the strategy should be calibrated for. Assumes
      (k, b)-fixed-epoch order participation.
    bands: The number of bands in the strategy.
    equal_norm: Flag to indicate that each column of C should have equal_norm.
      Useful for BandMF.  If epochs=1, this flag is a no-op, as the returned
      strategy will be column normalized either way.
    A: The workload matrix (defaults to Prefix).
    max_optimizer_steps: The maximum number of LBFGS steps to take.
    callback: An optional callback function to monitor optimization progress.
      The default callback terminates the optimization early if the projected
      gradient is near-zero.

  Returns:
    The strategy matrix C that minimizes expected total squared error.
  """
  A = jnp.tri(n) if A is None else A
  mask = get_orthogonal_mask(n, epochs)
  if bands is not None:
    mask = mask * sensitivity.banded_symmetric_mask(n, bands)

  def loss_and_projected_grad(X):
    loss, dX = _mean_loss_and_gradient(X, A)
    if equal_norm:
      diag = 0
    else:
      # normalizes sum_i dX[i,i] = 0, where sum is taken over iterations a
      # single user can participate in.  This ensures that sum_i X[i,i]
      # remains equal to 1.
      dsum = jnp.diag(dX).reshape(epochs, -1).sum(axis=0) / epochs
      diag = jnp.diag(dX) - jnp.kron(jnp.ones(epochs), dsum)
    dX = dX.at[jnp.diag_indices(n)].set(diag)
    # sets dX[i,j] = 0 if i \neq j and a user can appear in both round i and j.
    # If banded constraints are given, sets dX[i,j] = 0 if |i-j| > # bands.
    return loss, dX * mask

  X = optimization.optimize(
      loss_and_projected_grad,
      jnp.eye(n, dtype=jnp.float64) / epochs,
      max_optimizer_steps=max_optimizer_steps,
      grad=True,
      callback=callback,
  )

  return strategy_from_X(X)
