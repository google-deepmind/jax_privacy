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

"""Library for working with general Toeplitz matrices.

In particular, lower-triangular Toeplitz matrices used in DP matrix
factorization algorithms.  Generally, these functions operate directly in terms
of the `n` Toeplitz coefficients, and hence can be much more efficient than
doing the calculations on the materialized `n**2` matrices.
"""

import concurrent
import dataclasses
import functools
from typing import Any, Callable, Protocol
import dp_accounting
import jax
import jax.numpy as jnp
from . import optimization
from . import sensitivity
from . import streaming_matrix

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def _l2_norm_squared(x: jax.Array):
  return jnp.inner(x, x)


def _reconcile(coef: jax.Array, n: int | None = None) -> tuple[jax.Array, int]:
  """Reconciles the Toeplitz coefficients with the matrix size.

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix C, that
      is, `coef` are the leading nonzero entries of C[:, 0]. C is of size n x n;
      if len(coef) < n, the remaining coefficients are assumed to be zero. If
      len(coef) > n, then only the first n coefficients are used.
    n: Optional, the size of the matrix C (see `coef` above). If None, the size
      of the matrix is equal to the number of coefficients.

  Returns:
    A tuple (coef, n) where n >= len(coef). If a non-None `n` keyword argument
    was provided, then this function will return the same `n` value.
  """
  n = n or len(coef)
  coef = jnp.array(coef)[:n]  # Drop extra coefficients if needed.
  return coef, n


def pad_coefs_to_n(coef: jax.Array, n: int | None = None) -> jax.Array:
  """Materializes length-n Toeplitz coefficients."""
  coef, n = _reconcile(coef, n)
  return jnp.zeros(n).at[0 : len(coef)].set(coef)


def inverse_as_streaming_matrix(
    coef: jax.Array,
    column_normalize_for_n: int | None = None,
) -> streaming_matrix.StreamingMatrix:
  """Create $C^{-1}$ as a StreamingMatrix object.

  If column_normalize_for_n is None, the returned object represents $C^{-1}$,the
  inverse of an arbitrarily large banded Toeplitz matrix $C$ with coefficients
  `coef`.

  If column_normalize_for_n is finite, the returned object represents
  $C^{-1}$, the inverse of a banded matrix $C$ of the given size n x n, formed
  by taking the banded Toeplitz matrix with coefficients `coef` and re-scaling
  each column so it has L2 norm 1.0. We recommend setting column_normalize_for_n
  in centralized training scenarios where `n` is known in advance. The supplied
  `coef` must have an L2 norm of 1.0 in this case. Without this check, if
  `coef` had say an L2 norm of 0.5 and privacy calculations were done based on
  this, then column normalization would increase
  the sensitivity by a factor of 2 and the actual DP guarantee would be worse
  than calculated.

  TODO: b/409863240 - Consider other approaches to avoid this issue, and provide
  a method for computing the sensitivity of column-normalized Toeplitz matrices.

  Note: If the supplied `coef` do not have an L2 norm of 1.0, then specifying
  column_normalize_for_n may change the sensitivity of the implied $C$ matrix.

  This implementation is based on Algorithm 9 from
  https://arxiv.org/abs/2306.08153.

  Args:
    coef: The Toeplitz coefficients of the strategy.
    column_normalize_for_n: If given, the returned object represents the inverse
      of a *column-normalized* banded Toeplitz matrix of the given size.
      Otherwise, it reprsents the inverse of an ordinary banded Toeplitz matrix.
      If not None, the supplied `coef` must have an L2 norm of 1.0 (otherwise,
      column normalization could change the sensitivity of the implied $C$
      matrix).

  Returns:
    A StreamingMatrix object representing $C^{-1}$.
  """
  coef, _ = _reconcile(coef, column_normalize_for_n)
  bands = coef.shape[0]

  def init_fn(shape):
    return jnp.zeros((bands - 1,) + shape, dtype=coef.dtype)

  def next_fn(yi, state):
    if bands == 1:
      return yi / coef[0], state
    inner = jnp.tensordot(coef[1:], state, axes=1)
    xi = (yi - inner) / coef[0]
    return xi, jnp.roll(state, 1, axis=0).at[0].set(xi)

  C_inv = streaming_matrix.StreamingMatrix(init_fn, next_fn)

  if column_normalize_for_n is not None:
    coef_norm = jnp.linalg.norm(coef)
    if jnp.abs(coef_norm - 1) > 1e-6:
      raise ValueError(
          'If column_normalize_for_n is specified, then the supplied `coef`'
          f' must have an L2 norm of 1.0, but found norm {coef_norm}.'
      )
    # 1/s scale on the cols of C translates to scale of s on the rows of C^{-1}.
    col_norms = jnp.sqrt(
        jnp.cumsum(jnp.pad(coef**2, (0, column_normalize_for_n - bands)))[::-1]
    )
    C_inv = streaming_matrix.scale_rows_and_columns(C_inv, row_scale=col_norms)

  return C_inv


def optimal_max_error_strategy_coefs(n: int) -> jax.Array:
  """Returns the coefs of the optimal Toeplitz strategy matrix C for max error.

  These coefficients were introduced by Fichtenberger, Henzinger, and Upadhyay
  in "Constant Matters: Fine-grained Error Bound on Differentially Private
  Continual Observation"
  (https://proceedings.mlr.press/v202/fichtenberger23a/fichtenberger23a.pdf,
  https://arxiv.org/pdf/2202.11205),
  and proved to be optimal for max error under single participations by
  Dvijotham, McMahan, Pillutla, Steinke, and Thakurta in
   in "Efficient and Near-Optimal Noise Generation for Streaming Differential
   Privacy" (https://arxiv.org/abs/2404.16706).

  Args:
    n: The number of coefficients to return.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix C that
    factorizes the prefix sum matrix A as A = C @ C.
  """
  k = jnp.arange(n)
  return jnp.cumprod(((2 * k - 1) / (2 * k)).at[0].set(1))


def optimal_max_error_noising_coefs(n: int) -> jax.Array:
  """Returns the coefs of the optimal Toeplitz noise matrix for max error.

  Args:
    n: The number of coefficients to return.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix C^{-1} that
    is the inverse of the matrix returned by `optimal_max_error_strategy_coefs`.
  """
  # This factorization of A = B C where A is the prefix-sum matrix is symmetric,
  # in that C = B = A C^{-1}, so C^{-1} = A^{-1} C, where A^{-1}
  # computes differences.
  c = optimal_max_error_strategy_coefs(n)
  return c.at[1:n].subtract(c[:-1])


def materialize_lower_triangular(
    coef: jax.Array, n: int | None = None
) -> jax.Array:
  """Creates a lower-triangular Toeplitz matrix.

   Example: If `coef = [a, b, c]` and `n = 6`, then this method returns:

  ```
  [a 0 0 0 0 0]
  [b a 0 0 0 0]
  [c b a 0 0 0]
  [0 c b a 0 0]
  [0 0 c b a 0]
  [0 0 0 c b a]
  ```

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix C, that
      is, `coef` are the leading nonzero entries of C[:, 0]. C is of size n x n;
      if len(coef) < n, the remaining coefficients are assumed to be zero. If
      len(coef) > n, then only the first n coefficients are used.
    n: Optional, the size of the matrix C (see `coef` above). If None, the size
      of the matrix is equal to the number of coefficients.

  Returns:
    The lower-triangular Toeplitz matrix.
  """
  coef = pad_coefs_to_n(coef, n)
  return jax.scipy.linalg.toeplitz(coef, r=jnp.zeros_like(coef))


def solve_banded(coef: jax.Array, rhs: jax.Array) -> jax.Array:
  """Solve the linear system T_{coef} x = rhs for x for Toeplitz matrix T.

  Specifically, T_{coef} is a lower triangular banded Toeplitz matrix.

  Note we want to be able to back-propagate gradients through this function,
  hence we cannot use scipy.linalg.solve_toeplitz.

  Example: coef = [a, b, c], rhs = [1, 1, 1, 1, 1, 1], we solve the following
  system for x
  ```
  [a 0 0 0 0 0] [x_0]   [1]
  [b a 0 0 0 0] [x_1]   [1]
  [c b a 0 0 0] [x_2] = [1]
  [0 c b a 0 0] [x_3]   [1]
  [0 0 c b a 0] [x_4]   [1]
  [0 0 0 c b a] [x_5]   [1]
  ```

  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix C, that
      is, `coef` are the leading nonzero entries of C[:, 0]. C is of size n x n
      where n = len(rhs) (see below); if len(coef) < n, the remaining
      coefficients are assumed to be zero. If len(coef) > n, then only the first
      n coefficients are used.
    rhs: The right hand side vector, of length `n`.

  Returns:
    The solution to the linear system Toeplitz(coef, n) x = rhs.
  """
  return inverse_as_streaming_matrix(coef) @ rhs


def multiply(
    lhs_coef: jax.Array,
    rhs_coef: jax.Array,
    n: int | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Computes the matrix product of two lower-triangular Toeplitz matrices.

  Args:
    lhs_coef: The nonzero coefficients of a lower-triangular Toeplitz matrix L,
      that is, `lhs_coef` are the leading nonzero entries of L[:, 0]. L is of
      size n x n; if len(lhs_coef) < n, the remaining coefficients are assumed
      to be zero. If len(lhs_coef) > n, then only the first n coefficients are
      used.
    rhs_coef: The nonzero coefficients of a lower-triangular Toeplitz matrix R,
      under the same conventions as `lhs_coef`.
    n: Optional, the size of the matrices L and R (see `coef` above). If None,
      the size of the matrices is equal to the number of coefficients.
    skip_checks: If True, don't perform input verification. Setting `skip_checks
      = True` is necessary when this function is jitted.

  Returns:
    The coefficients of the lower-triangular Toeplitz matrix L @ R where
    L = materialize_lower_triangular(lhs_coef, n) and
    R = materialize_lower_triangular(rhs_coef, n).
  """
  if not skip_checks:
    if n is None and len(lhs_coef) != len(rhs_coef):
      raise ValueError(
          'If n is not specified, then lhs_coef and rhs_coef must have the same'
          ' length, but found lhs_coef of length'
          f' {len(lhs_coef)=} and rhs_coef of length {len(rhs_coef)=}.'
      )
  lhs_coef, n = _reconcile(lhs_coef, n)
  rhs_coef, _ = _reconcile(rhs_coef, n)
  return jnp.convolve(
      lhs_coef, rhs_coef, mode='full', precision=jax.lax.Precision.HIGHEST
  )[:n]


def inverse_coef(coef: jax.Array, n: int | None = None) -> jax.Array:
  """Finds the inverse coefficients of a lower-triangularToeplitz matrix.

  If C is a lower-triangular Toeplitz matrix, then so is C^{-1}; this function
  returns the Toeplitz coefficients of this inverse.


  Args:
    coef: The nonzero coefficients of a lower-triangular Toeplitz matrix C, that
      is, `coef` are the leading nonzero entries of C[:, 0]. C is of size n x n;
      if len(coef) < n, the remaining coefficients are assumed to be zero. If
      len(coef) > n, then only the first n coefficients are used.
    n: Optional, the size of the matrix C (see `coef` above). If None, the size
      of the matrix is equal to the number of coefficients.

  Returns:
    The Toeplitz coefficients of C^{-1}, of length n.
  """
  coef, n = _reconcile(coef, n)
  return solve_banded(coef, jnp.zeros(n).at[0].set(1))


def sensitivity_squared(coef: jax.Array, n: int | None = None) -> jax.Array:
  """Sensitivity^2 under single participation."""
  coef, _ = _reconcile(coef, n)
  return _l2_norm_squared(coef)


def minsep_sensitivity_squared(
    strategy_coef: jax.Array,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Returns the sensitivity of the Toeplitz matrix.

  With max_participations = 1 (and any min_sep, say min_sep = 1), this is the
  same as single participation.

  Reference: While this code actually predates the paper, this result is
  published in https://arxiv.org/pdf/2405.13763, Theorem 2.

  Args:
    strategy_coef: The nonzero coefficients of the Toeplitz matrix C used in the
      matrix mechanism with factorization A = B C. That is, `coef` are the
      leading nonzero entries of C[:, 0]. C is of size n x n; if len(coef) < n,
      the remaining coefficients are assumed to be zero. If len(coef) > n, then
      only the first n coefficients are used.
    min_sep: The minimum separation between two participation of a worst-case
      client/sample. Note that we use the definition in [(Amplified) Banded
      Matrix Factorization: A unified approach to private
      training](https://arxiv.org/abs/2306.08153). For a user participating on
      iteration $i$ and then again on iteration $j$,  the separation is $j -i$;
      that is, a min_sep of 1 allows participation on every iteration.
    max_participations: The maximum participation of a worst-case user.
    n: Optional, the size of the matrix C (see `coef` above). If None, the size
      of the matrix is equal to the number of coefficients.
    skip_checks: If True, don't perform input verification which may not be
      supported in jitted contexts.

  Returns:
    The sensitivity squared.
  """
  coef, n = _reconcile(strategy_coef, n)

  # We may need to turn these off in jitted contexts:
  if not skip_checks:
    if not jnp.all(coef >= 0):
      raise ValueError(
          f'coef must be non-negative, but found minimum value {jnp.min(coef)},'
          f' {coef[:25]=}'
      )
    if len(coef) > 1:
      incr = coef[1:] - coef[:-1]
      max_incr = jnp.max(incr)
      if max_incr > 0:
        raise ValueError(
            f'coef must be non-increasing, but found increase {max_incr} at'
            f' index {jnp.argmax(incr)}'
        )
    if not min_sep > 0:
      raise ValueError('min_sep must be positive')

  k = sensitivity.minsep_true_max_participations(
      n=n, min_sep=min_sep, max_participations=max_participations
  )
  # Because we assume the Toeplitz coefficients are positive and decreasing,
  # the worst-case for sensitivity is (up to) k participations separated by
  # exactly b.  We use a difference of cumsums to do this in O(n) time.

  padding = (min_sep - n) % min_sep
  coef = jnp.pad(coef, (0, n - coef.size + padding))
  vector = coef.reshape(-1, min_sep).cumsum(axis=0).flatten()
  vector = vector.at[min_sep * k :].set(
      vector[min_sep * k :] - vector[: -min_sep * k]
  )
  return vector[:n] @ vector[:n]


def per_query_error(
    *,
    strategy_coef: jax.Array | None = None,
    noising_coef: jax.Array | None = None,
    n: int | None = None,
    workload_coef: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Expected per-query squared error for a (banded) Toeplitz mechanism.

  Exactly one of `strategy_coef` and `noising_coef` should be provided.

  Args:
    strategy_coef: Toeplitz coefficients of the strategy matrix.
    noising_coef: Toeplitz coefficients of the noising matrix.
    n: The size of the implied matrices (defaults to the length of the Toeplitz
      coefficient array).
    workload_coef: Toeplitz coefficients of the workload matrix. Defaults to the
      vector of 1s, corresponding to the prefix matrix. If this is longer than
      `n`, the extra entries are ignored (even if `n` is inferred from the
      length of the `strategy_coef` or `noising_coef`).
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected per-query squared error, an array of length n.
  """
  if not skip_checks:
    if (strategy_coef is None) == (noising_coef is None):
      raise ValueError('Specify exactly one of strategy_coef or noising_coef.')

  if strategy_coef is not None:
    strategy_coef, n = _reconcile(strategy_coef, n)
    if workload_coef is not None:
      workload_coef = pad_coefs_to_n(workload_coef, n)
    else:
      workload_coef = jnp.ones(n)
    B_coef = solve_banded(strategy_coef, workload_coef)
  else:
    assert noising_coef is not None
    noising_coef, n = _reconcile(noising_coef, n)
    if workload_coef is None:
      # This is more efficient than explicitly multiplying by the prefix matrix.
      B_coef = jnp.cumsum(noising_coef)
    else:
      B_coef = multiply(
          workload_coef, noising_coef, n=n, skip_checks=skip_checks
      )

  return jnp.cumsum(B_coef**2)


def max_error(
    *,
    strategy_coef: jax.Array | None = None,
    noising_coef: jax.Array | None = None,
    n: int | None = None,
    workload_coef: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Max-over-iterations squared error for a (banded) Toeplitz mechanism.

  Exactly one of `strategy_coef` and `noising_coef` should be provided.

  Args:
    strategy_coef: Toeplitz coefficients of the strategy matrix.
    noising_coef: Toeplitz coefficients of the noising matrix.
    n: The size of the implied matrices (defaults to the length of the Toeplitz
      coefficient array).
    workload_coef: Toeplitz coefficients of the workload matrix. Defaults to the
      vector of 1s, corresponding to the prefix matrix.
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected max-over-iterations squared error.
  """
  # It is a special property of Toeplitz matrices that the max error always
  # occurs on the last iteration. Thus, for max_error materializing the full
  # per_query_error is not necessary. However, benchmarking indicates
  # that this is not a significant performance hit. In particular,
  # when computing max_error given noising_coefs (and so not including the
  # expensive inverse calculation), with n=100_000 on CPU
  # we see this is about 7% slower, and on GPUs there is no performance
  # penalty (possibly due to better compiler optimizations?).
  return per_query_error(
      strategy_coef=strategy_coef,
      noising_coef=noising_coef,
      n=n,
      workload_coef=workload_coef,
      skip_checks=skip_checks,
  )[-1]


def mean_error(
    *,
    strategy_coef: jax.Array | None = None,
    noising_coef: jax.Array | None = None,
    n: int | None = None,
    workload_coef: jax.Array | None = None,
    skip_checks: bool = False,
) -> jax.Array:
  """Mean-over-iterations squared error for a (banded) Toeplitz mechanism.

  Exactly one of `strategy_coef` and `noising_coef` should be provided.

  Args:
    strategy_coef: Toeplitz coefficients of the strategy matrix.
    noising_coef: Toeplitz coefficients of the noising matrix.
    n: The size of the implied matrices (defaults to the length of the Toeplitz
      coefficient array).
    workload_coef: Toeplitz coefficients of the workload matrix. Defaults to the
      vector of 1s, corresponding to the prefix matrix.
    skip_checks: If True, don't perform input verification. It may be necessary
      to set skip_checks=True when this function is jitted.

  Returns:
    The expected mean-over-iterations squared error.
  """
  return jnp.mean(
      per_query_error(
          strategy_coef=strategy_coef,
          noising_coef=noising_coef,
          n=n,
          workload_coef=workload_coef,
          skip_checks=skip_checks,
      )
  )


class ErrorOrLossFn(Protocol):
  """Protocol for error functions used in `loss()` below."""

  def __call__(
      self,
      *,
      strategy_coef: jax.Array,
      n: int | None = None,
  ) -> jax.Array:
    ...


@functools.partial(jax.jit, static_argnums=[1, 2])
def loss(
    strategy_coef: jax.Array,
    n: int | None = None,
    error_fn: ErrorOrLossFn = mean_error,
) -> jax.Array:
  """Error of C on prefix workload under single participation.

  See Scaling up the Amplified Banded Matrix Factorization Mechanism for
  Differentially Private ML (https://arxiv.org/abs/2405.15913) for details.

  Args:
    strategy_coef: The nonzero coefficients of the Toeplitz matrix C used in the
      matrix mechanism with factorization A = B C. That is, `coef` are the
      leading nonzero entries of C[:, 0]. C is of size n x n; if len(coef) < n,
      the remaining coefficients are assumed to be zero. If len(coef) > n, then
      only the first n coefficients are used.
    n: Optional, the size of the matrix C (see `coef` above). If None, the size
      of the matrix is equal to the number of coefficients.
    error_fn: The objective function to use (e.g., mean_error or max_error).

  Returns:
    The total squared error times sensitivity of the toeplitz C on the
    prefix-sum workload under single participation.
  """
  # This is the maximum column norm of C, i.e., single-epoch sensitivity.
  strategy_coef, n = _reconcile(strategy_coef, n)
  error = error_fn(strategy_coef=strategy_coef, n=n)
  sens_squared = sensitivity_squared(strategy_coef, n)
  return error * sens_squared


mean_loss = functools.partial(loss, error_fn=mean_error)
max_loss = functools.partial(loss, error_fn=max_error)


def optimize_banded_toeplitz(
    n: int,
    bands: int,
    strategy_coef: jax.Array | None = None,
    max_optimizer_steps: int = 250,
    loss_fn: ErrorOrLossFn = mean_loss,
) -> jax.Array:
  """Optimize over the space of banded Toeplitz strategies on a Prefix workload.

  The banded toeplitz strategies produced by this function can be used for
  both the single-participation setting and the multi-participation setting,
  (including both the `fixed_epoch_order` and `min_sep` participation schemas;
  see README.md) as long as the (minimum) separation between contributions from
  the same user is at least the number of bands provided.
  See https://arxiv.org/abs/2306.08153 for more details.

  If used with a different participation pattern (e.g., (k, b)-minsep where
  b is less than the number of bands, sensitivity can be computed post-hoc
  using e.g. `toeplitz.minsep_sensitivity_squared`.  This should not be
  necessary in centralized training regimes where the exact participation
  pattern should be known in advance, however.

  Args:
    n: the number of iterations that defines the workload.
    bands: The number of bands in the Toeplitz matrix.
    strategy_coef: Optional toeplitz coefficients to initialize optimization.
    max_optimizer_steps: The maximum number of LBFGS iterations.
    loss_fn: The loss function to use (e.g., mean_loss or max_loss). Should
      consume `coefs` with len(coefs) == bands and `n` as arguments.

  Returns:
    The coefficeints of the optimal banded Toeplitz strategy, guaranteed to
    have L2 norm 1.
  """
  loss_fn = functools.partial(loss_fn, n=n)

  if strategy_coef is None:
    strategy_coef = optimal_max_error_strategy_coefs(bands)
  if strategy_coef.shape[0] != bands:
    raise ValueError(f'{strategy_coef.shape=} != {bands=}')

  params = optimization.optimize(
      loss_fn, strategy_coef, max_optimizer_steps=max_optimizer_steps
  )
  return params / jnp.linalg.norm(params)


def _factors(n):
  result = functools.reduce(
      list.__add__,
      ([i, n // i] for i in range(1, int(jnp.sqrt(n) + 1)) if n % i == 0),
  )
  # Return a sorted list, remove duplicates.
  return list(sorted(set(result)))


@dataclasses.dataclass(frozen=True)
class _AmplifiedBandMFHelper:
  """A convienence class for building an amplified BandMF mechanism.

  This class is primarily used to implement
  `optimize_banded_toeplitz_for_amplifications` below.
  """

  n: int
  dataset_size: int
  batch_size: int  # The (expected) batch size.

  epsilon: float
  delta: float

  loss_fn: ErrorOrLossFn = max_loss
  make_fresh_accountant: Callable[[], dp_accounting.PrivacyAccountant] = (
      dp_accounting.rdp.RdpAccountant
  )

  @property
  def max_bands(self):
    """The largest number of bands b possible while achieving b-min-sep."""
    return self.dataset_size // self.batch_size

  def sensitivity(self, coef: jax.Array):
    """The sensitivity of the banded Toeplitz matrix C."""
    return float(jnp.sqrt(sensitivity_squared(coef, n=self.n)))

  def total_noise_multiplier(self, bands: int):
    """The total noise multiplier needed to achieve the privacy target."""
    # It is preferable if n % bands == 0.
    max_participations = int(jnp.ceil(self.n / bands))

    # It is also preferable if dataset_size % bands == 0.
    subset_size = self.dataset_size // bands
    sampling_probability = self.batch_size / subset_size
    max_noise_multiplier = int(100 * jnp.sqrt(self.batch_size))

    def dpsgd_event(noise_multiplier):
      one_round_event = dp_accounting.PoissonSampledDpEvent(
          sampling_probability=sampling_probability,
          event=dp_accounting.GaussianDpEvent(
              noise_multiplier=noise_multiplier
          ),
      )
      return dp_accounting.SelfComposedDpEvent(
          event=one_round_event, count=max_participations
      )

    return dp_accounting.calibrate_dp_mechanism(
        # ADD_OR_REMOVE_ONE is the default neighboring relation.
        make_fresh_accountant=self.make_fresh_accountant,
        make_event_from_param=dpsgd_event,
        target_epsilon=self.epsilon,
        target_delta=self.delta,
        # We have to hack things a little bit so RDP accounting doesn't blow up.
        bracket_interval=dp_accounting.ExplicitBracketInterval(
            0.01, max_noise_multiplier
        ),
    )

  def required_stddev(self, coef: jax.Array):
    """The stddev of the uncorrelated noise Z required.

    That is, passing this stddev to
    `distributed_noise_generation.streaming_matrix_to_single_machine_privatizer`
    should achieve the (epsilon, delta)-DP guarantee.

    Args:
      coef: The coefficients of the banded Toeplitz matrix C.

    Returns:
      The stddev of the uncorrelated noise Z required.
    """
    total_nm = self.total_noise_multiplier(bands=len(coef))
    return total_nm * self.sensitivity(coef)

  def amplified_bandmf_loss(self, coef: jax.Array):
    """The loss in the estimate of the *average* prefix sum."""
    error_times_sens = jnp.sqrt(self.loss_fn(strategy_coef=coef, n=self.n))
    # Note: loss = error * single_participation_sensitivity
    # We would normally take
    #  stddev = total_nm * single_participation_sensitivity
    #  loss = error * stddev / batch_size
    # The following is equivalent.
    total_nm = self.total_noise_multiplier(bands=len(coef))
    return error_times_sens * total_nm / self.batch_size

  def compute_loss_for_bands(
      self,
      bands_list: list[int] | None = None,
      max_workers: int | None = None,
      **optimizer_kwargs,
  ) -> list[dict[str, Any]]:
    """Computes the loss for each value in a list of possible bands.

    Args:
      bands_list: The list of possible bands to compute the loss for. If None,
        all factors of `n` less than the maximum number of possible bands are
        considered.
      max_workers: The maximum number of workers to use, passed to the
        ThreadPoolExecutor.
      **optimizer_kwargs: Keyword arguments passed to
        `optimize_banded_toeplitz`.

    Returns:
      A list of dictionaries, one for each value in `bands_list`. Each
      dictionary contains the following keys:
        - `bands`: The number of bands.
        - `coef`: The coefficients of the banded Toeplitz matrix C.
        - `loss`: The loss of the estimate of the *average* prefix sum.
    """
    assert 1 <= self.batch_size <= self.dataset_size
    if bands_list is None:
      bands_list = [
          b for b in _factors(self.dataset_size) if b <= self.max_bands
      ]

    if max(bands_list) > self.max_bands:
      raise ValueError(
          f'bands_list={bands_list} contains a value that exceeds '
          f'max_bands={self.max_bands}'
      )

    def run_task(b):
      coef = optimize_banded_toeplitz(
          n=self.n, bands=b, loss_fn=self.loss_fn, **optimizer_kwargs
      )
      loss_value = self.amplified_bandmf_loss(coef)
      return {'bands': b, 'coef': coef, 'loss': loss_value}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      return list(executor.map(run_task, bands_list))

  def optimize_bands(self, **optimizer_kwargs) -> dict[str, Any]:
    """Returns best result (in terms of loss) from compute_loss_for_bands."""
    results = self.compute_loss_for_bands(**optimizer_kwargs)
    losses = jnp.array([d['loss'] for d in results])
    best_idx = jnp.argmin(losses)
    return results[best_idx]


def optimize_coefs_for_amplifications(
    n: int,
    *,
    dataset_size: int,
    expected_batch_size: int,
    epsilon: float,
    delta: float,
    max_optimizer_steps: int = 250,
    loss_fn: ErrorOrLossFn = mean_loss,
) -> tuple[jax.Array, float]:
  """Select num_bands (and coefs) to minimize loss subject to a privacy target.

  Following Theorem 4 of https://arxiv.org/abs/2306.08153, this function
  (approximately) minimizes the loss_fn assuming privacy amplification under
  block-cyclic Poisson sampling (Algorithm 2 of
  https://arxiv.org/abs/2306.08153). A smaller number of bands allows more
  benefit from amplification, while a larger number of bands allows more benefit
  from correlated noise.

  Notes:
   - This function only optimizes over numbers of bands that evenly divide `n`,
      as this is generally preferable. Hence, it is recommended to choose `n` so
      it has well spaced factors; powers of 2 are particularly useful.
   - This function delegates to `optimize_banded_toeplitz` to actually
      optimize for the coefficients at a given number of bands. Hence, column
      normalization is not directly supported, but the final returned strategy
      can always be used with column normalization.

  Args:
    n: the number of iterations that defines the workload.
    dataset_size: The size of the dataset.
    expected_batch_size: The target batch size (so for example if we were
      Poisson sampling from the whole dataset, the sampling probability would be
      `expected_batch_size / dataset_size`).
    epsilon: The privacy target is (epsilon, delta)-DP.
    delta: The privacy target is (epsilon, delta)-DP.
    max_optimizer_steps: The maximum number of LBFGS iterations, passed to
      `optimize_banded_toeplitz`.
    loss_fn: The loss function to use (e.g., mean_loss or max_loss), passed to
      `optimize_banded_toeplitz`.

  Returns:
    A tuple `(coefs, stddev)` where:
      - `coefs` are the coefficeints of a banded Toeplitz strategy; the number
         of bands chosen is simply the length of the returned coefficients.
      - `stddev` is the stddev of the uncorrelated noise Z required to achieve
        the privacy target (that, is, passing this stddev to
        `streaming_matrix_to_single_machine_privatizer` in
        `distributed_noise_generation` should achieve the (epsilon, delta)-DP
        guarantee).
  """
  helper = _AmplifiedBandMFHelper(
      n, dataset_size, expected_batch_size, epsilon, delta, loss_fn
  )
  coef = helper.optimize_bands(max_optimizer_steps=max_optimizer_steps)['coef']
  stddev = helper.required_stddev(coef)
  return coef, stddev
