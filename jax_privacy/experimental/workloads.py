# Copyright 2026 DeepMind Technologies Limited.
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

"""Workload definitions for DP matrix factorization.

A workload matrix A defines the set of linear queries that the matrix mechanism
should answer accurately.  Different workloads support different efficient
representations:

- ``dense``: A full materialized ``jax.Array`` (universal fallback).
- ``streaming_matrix``: A ``StreamingMatrix`` for O(1)-memory multiplication.
- ``toeplitz_coefs``: Toeplitz coefficient array for O(n) arithmetic.
- ``gram``: The Gram matrix ``A^T A`` used in error computation.

Atomic workloads (``PrefixSum``, ``SuffixSum``, ``Identity``, ``AllRange``,
etc.) can be composed via:

- ``ScalarScaled(alpha, base)``: scalar multiplication ``alpha * A``.
- ``DiagonallyScaled(scale, base)``: left-multiplication by ``diag(scale)``.
- ``Stacked(w1, w2, ...)``: vertical stacking ``[A1; A2; ...]``.

Example usage::

  # Regularised prefix-sum workload [A; lambda * I].
  w = Stacked(PrefixSum(n=1000), ScalarScaled(0.1, Identity(n=1000)))
  A = w.dense          # shape (2000, 1000)
  G = w.gram           # shape (1000, 1000), computed as sum of sub-grams
"""

from __future__ import annotations

import abc
import dataclasses

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import streaming_matrix

# pylint:disable=invalid-name


class Workload(abc.ABC):
  """A workload matrix A for the matrix mechanism.

  Each workload stores the matrix dimension ``n`` and exposes representations
  as properties.  Not all representations are available for all workloads;
  ``streaming_matrix`` and ``toeplitz_coefs`` return ``None`` when the workload
  cannot be efficiently represented in that format.

  Subclasses must provide ``n`` (as a field or property) and implement
  ``dense``.  All other properties have sensible defaults.
  """

  n: int

  @property
  @abc.abstractmethod
  def dense(self) -> jax.Array:
    """Materialize as a dense matrix of shape ``(num_queries, n)``."""

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix | None:
    """Streaming representation, or ``None`` if unavailable."""
    return None

  @property
  def toeplitz_coefs(self) -> jax.Array | None:
    """Toeplitz coefficients, or ``None`` if not representable."""
    return None

  @property
  def gram(self) -> jax.Array:
    """Gram matrix ``A^T A`` of shape ``(n, n)``."""
    A = self.dense
    return A.T @ A


# ---------------------------------------------------------------------------
# Atomic workloads
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PrefixSum(Workload):
  """The prefix-sum (running total) workload: ``A = jnp.tri(n)``."""

  n: int

  @property
  def dense(self) -> jax.Array:
    return jnp.tri(self.n)

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix:
    return streaming_matrix.prefix_sum()

  @property
  def toeplitz_coefs(self) -> jax.Array:
    return jnp.ones(self.n)


@dataclasses.dataclass(frozen=True)
class SuffixSum(Workload):
  """The suffix-sum workload: reversed prefix-sum."""

  n: int

  @property
  def dense(self) -> jax.Array:
    return jnp.tri(self.n)[::-1, ::-1]


@dataclasses.dataclass(frozen=True)
class Identity(Workload):
  """The identity workload (per-step accuracy): ``A = jnp.eye(n)``."""

  n: int

  @property
  def dense(self) -> jax.Array:
    return jnp.eye(self.n)

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix:
    return streaming_matrix.identity()

  @property
  def toeplitz_coefs(self) -> jax.Array:
    return jnp.zeros(self.n).at[0].set(1.0)

  @property
  def gram(self) -> jax.Array:
    return jnp.eye(self.n)


@dataclasses.dataclass(frozen=True)
class AllRange(Workload):
  """All contiguous range queries.  Shape is ``(n*(n+1)/2, n)``."""

  n: int

  @property
  def dense(self) -> jax.Array:
    n = self.n
    # Prefix-sum rows: P[k] has ones in positions 0..k.
    # Prepend a zero row so that P_ext[0] = 0, P_ext[k] = P[k-1].
    P_ext = jnp.concatenate([jnp.zeros((1, n)), jnp.tri(n)], axis=0)
    # All (i, j) pairs with 0 <= i <= j < n.
    i_idx, j_idx = jnp.triu_indices(n)
    return P_ext[j_idx + 1] - P_ext[i_idx]

  @property
  def gram(self) -> jax.Array:
    """Efficient O(n²) gram via the min-outer-product identity."""
    r = jnp.arange(self.n) + 1
    X = jnp.outer(r, r[::-1])
    return jnp.minimum(X, X.T)


@dataclasses.dataclass(frozen=True)
class DenseWorkload(Workload):
  """A workload defined by an explicit dense matrix.

  Attributes:
    matrix: The workload matrix of shape ``(num_queries, n)``.
  """

  matrix: jax.Array

  @property
  def n(self) -> int:
    """Number of columns, inferred from the matrix shape."""
    return self.matrix.shape[1]

  @property
  def dense(self) -> jax.Array:
    return self.matrix


@dataclasses.dataclass(frozen=True)
class MomentumCooldown(Workload):
  """Momentum SGD with a learning-rate schedule.

  Attributes:
    momentum: The momentum coefficient (e.g. 0.9).
    learning_rates: Per-step learning rates, shape ``(n,)``.
    n: Number of time-steps.
  """

  momentum: float
  learning_rates: jax.Array
  n: int

  @property
  def dense(self) -> jax.Array:
    return self.streaming_matrix.materialize(self.n)

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix:
    return streaming_matrix.momentum_sgd_matrix(
        self.momentum, self.learning_rates
    )


# ---------------------------------------------------------------------------
# Composite workloads
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScalarScaled(Workload):
  """Scalar-scaled workload: ``alpha * A``.

  Attributes:
    alpha: The scalar multiplier.
    base: The base workload to scale.
  """

  alpha: float
  base: Workload

  @property
  def n(self) -> int:
    """Inherited from the base workload."""
    return self.base.n

  @property
  def dense(self) -> jax.Array:
    return self.alpha * self.base.dense

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix | None:
    base_sm = self.base.streaming_matrix
    if base_sm is None:
      return None
    return base_sm * self.alpha

  @property
  def toeplitz_coefs(self) -> jax.Array | None:
    c = self.base.toeplitz_coefs
    return self.alpha * c if c is not None else None

  @property
  def gram(self) -> jax.Array:
    return self.alpha**2 * self.base.gram


@dataclasses.dataclass(frozen=True)
class DiagonallyScaled(Workload):
  """Diagonally-scaled workload: ``diag(scale) @ A``.

  Attributes:
    scale: A 1-D array of per-query scale factors, length ``num_queries``.
    base: The base workload to scale.

  Raises:
    ValueError: If ``len(scale)`` does not match ``base.dense.shape[0]``.
  """

  scale: jax.Array
  base: Workload

  def __post_init__(self):
    # Validate scale length against number of queries.
    num_queries = self.base.dense.shape[0]
    if self.scale.shape[0] != num_queries:
      raise ValueError(
          f'scale length {self.scale.shape[0]} != '
          f'number of queries {num_queries}'
      )

  @property
  def n(self) -> int:
    """Inherited from the base workload."""
    return self.base.n

  @property
  def dense(self) -> jax.Array:
    return self.scale[:, None] * self.base.dense

  @property
  def streaming_matrix(self) -> streaming_matrix.StreamingMatrix | None:
    base_sm = self.base.streaming_matrix
    if base_sm is None:
      return None
    return streaming_matrix.scale_rows_and_columns(
        base_sm, row_scale=self.scale
    )

  @property
  def toeplitz_coefs(self) -> jax.Array | None:
    # Diagonal scaling breaks Toeplitz structure.
    return None


@dataclasses.dataclass(frozen=True)
class Stacked(Workload):
  """Vertically stacked workloads: ``[A1; A2; ...]``.

  All sub-workloads must have the same ``n``.

  Attributes:
    workloads: The tuple of workloads to stack vertically.

  Raises:
    ValueError: If fewer than 2 workloads are provided or ``n`` values
      don't match across sub-workloads.
  """

  workloads: tuple[Workload, ...]

  def __init__(self, *workloads: Workload):
    """Validates and stores the sub-workloads."""
    if len(workloads) < 2:
      raise ValueError('Stacked requires at least 2 workloads.')
    ns = {w.n for w in workloads}
    if len(ns) > 1:
      raise ValueError(f'All stacked workloads must have the same n, got {ns}')
    object.__setattr__(self, 'workloads', workloads)

  @property
  def n(self) -> int:
    """Inherited from the first sub-workload."""
    return self.workloads[0].n

  @property
  def dense(self) -> jax.Array:
    return jnp.concatenate([w.dense for w in self.workloads], axis=0)

  @property
  def gram(self) -> jax.Array:
    """Sum of sub-workload grams: ``sum_i A_i^T A_i``."""
    return sum(w.gram for w in self.workloads)
