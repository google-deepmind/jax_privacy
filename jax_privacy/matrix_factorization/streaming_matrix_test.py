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

import itertools
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import streaming_matrix
import numpy as np
import scipy

# pylint:disable=invalid-name


# An example streaming matrix with non-trivial state.
def get_inverse_banded_matrix(n):
  return banded.ColumnNormalizedBanded.default(
      n, bands=min(4, n)
  ).inverse_as_streaming_matrix()


# A dict of names to constructors; we return constructor functions
# rather than objects to avoid "RuntimeError: Attempted call to JAX before
# absl.app.run() is called," as well as to also allow independent
# parameterization of tests over n. Note that most of these matrices `n`
# is not used, and all of them should still defind infinte-dimensional
# operators, that is, they can be materialized for any n > 0
TEST_MATRICES = {
    'prefix_sum': lambda n: streaming_matrix.prefix_sum(),
    'identity': lambda n: streaming_matrix.identity(),
    'momentum_sgd': lambda n: streaming_matrix.momentum_sgd_matrix(
        momentum=0.9, learning_rates=jnp.linspace(1.0, 0.1, num=n)
    ),
    'diagonal': lambda n: streaming_matrix.diagonal(jnp.arange(1, n + 1)),
    'inverse_banded': get_inverse_banded_matrix,
    'scaled_inv_banded': lambda n: (
        get_inverse_banded_matrix(n).scale_rows_and_columns(
            row_scale=jnp.arange(1, n + 1),
            col_scale=jnp.linspace(1.0, 0.1, num=min(n, 3)),
        )
    ),
}


def matrix_and_n_params(n_values: list[int] | None = None):
  """A list of (name, matrix_fn, n) tuples for parameterized tests."""
  n_values = n_values or [1, 2, 7]
  param_list = []
  for (A_name, A_fn), n in itertools.product(TEST_MATRICES.items(), n_values):
    param_list.append((f'{A_name}_{n=}', A_fn, n))
  return param_list


def matrix_pair_and_n_params(n_values: list[int] | None = None):
  """A list of (name, A_fn, B_fn, n) tuples for parameterized tests."""
  n_values = n_values or [1, 2, 7]
  param_list = []
  for (A_name, A_fn), (B_name, B_fn), n in itertools.product(
      TEST_MATRICES.items(), TEST_MATRICES.items(), n_values
  ):
    param_list.append((f'A={A_name}_B={B_name}_{n=}', A_fn, B_fn, n))
  return param_list


def assert_allclose(a, b):
  np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def _concrete_momentum_sgd_matrix(
    num_iters: int, momentum: float, learning_rates: Optional[np.ndarray] = None
) -> np.ndarray:
  """Returns a numpy array representing momentum SGD."""
  if learning_rates is None:
    learning_rates = np.ones(num_iters)

  # Banded matrix that computes the t'th value of the momentum buffer
  m_powers = [momentum**i for i in range(num_iters)]
  m_buf_matrix = scipy.sparse.diags(
      m_powers,
      offsets=-np.arange(num_iters, dtype=np.int32),
      shape=(num_iters, num_iters),
  ).toarray()

  # Lower triangular matrix with nonzeros in column i equal to learning_rates[i]
  lr_matrix = np.tri(num_iters) @ np.diag(learning_rates)
  return lr_matrix @ m_buf_matrix


class StreamingMatrixTest(parameterized.TestCase):

  def test_prefix_sum(self):
    z = jnp.arange(10)
    A = streaming_matrix.prefix_sum()
    exp = jnp.cumsum(z)
    assert_allclose(exp, A @ z)
    assert_allclose(A.materialize(10), jnp.tri(10))

  def test_identity(self):
    z = jnp.arange(10)
    A = streaming_matrix.identity()
    assert_allclose(z, A @ z)
    assert_allclose(A.materialize(10), jnp.eye(10))

  def test_diagonal(self):
    z = jnp.arange(10)
    A = streaming_matrix.diagonal(z)
    assert_allclose(z * z, A @ z)
    assert_allclose(A.materialize(10), jnp.diag(z))

  def test_momentum_cooldown(self):
    """Test MomentumCooldown."""
    z = jnp.ones(8)
    learning_rates = jnp.ones(8).at[7].set(0.1)
    A = streaming_matrix.momentum_sgd_matrix(
        momentum=1, learning_rates=learning_rates
    )
    exp = jnp.array([1, 3, 6, 10, 15, 21, 28, 28.8])
    assert_allclose(exp, A @ z)

  def test_momentum_cooldown_integration(self):
    """Test MomentumCooldown."""
    z = jax.random.normal(jax.random.PRNGKey(42), shape=(10,))
    learning_rates = jax.random.uniform(jax.random.PRNGKey(43), shape=(10,))
    A = streaming_matrix.momentum_sgd_matrix(
        momentum=0.9, learning_rates=learning_rates
    )
    exp = (
        _concrete_momentum_sgd_matrix(
            10, momentum=0.9, learning_rates=learning_rates
        )
        @ z
    )
    assert_allclose(exp, A @ z)

  def test_multiply_streaming_matrices(self):
    A = streaming_matrix.prefix_sum()
    z = jnp.arange(10)
    assert_allclose(A @ (A @ z), (A @ A) @ z)

  @parameterized.parameters(1, 10)
  def test_multiply_streaming_matrices_diagonal(self, n):
    d = jnp.linspace(1.0, 2.0, num=10)
    A = streaming_matrix.diagonal(d)
    B = streaming_matrix.diagonal(1 / d)
    AB = streaming_matrix.multiply_streaming_matrices(A, B)
    assert_allclose(AB.materialize(n), jnp.eye(n))

  @parameterized.named_parameters(matrix_pair_and_n_params())
  def test_multiply_vs_materialized(self, A, B, n):
    A, B = A(n), B(n)
    Am, Bm = A.materialize(n), B.materialize(n)
    assert_allclose(Am @ Bm, (A @ B).materialize(n))

  def test_row_norms_squared(self):
    d = jnp.linspace(1.0, 2.0, num=10)
    A = streaming_matrix.diagonal(d)
    assert_allclose(A.row_norms_squared(10), d**2)

    A = streaming_matrix.prefix_sum()
    assert_allclose(A.row_norms_squared(10), jnp.arange(1, 11))

  @parameterized.named_parameters(matrix_and_n_params())
  def test_row_norms_squared_vs_materialized(self, A, n):
    A = A(n)
    assert_allclose(
        A.row_norms_squared(n), jnp.linalg.norm(A.materialize(n), axis=1) ** 2
    )

    # All of our streaming should "extend" beyond the n for which they are
    # constructed, so test this as well:
    assert_allclose(
        A.row_norms_squared(n + 3),
        jnp.linalg.norm(A.materialize(n + 3), axis=1) ** 2,
    )

  def test_scale_rows_and_columns(self):
    n = 3
    M = streaming_matrix.prefix_sum()
    row_scale = jnp.array([1, 2, 3])
    col_scale = jnp.array([1, 2, 3])

    # Rows only:
    R = M.scale_rows_and_columns(row_scale=row_scale)
    assert_allclose(
        R.materialize(n), jnp.array([[1, 0, 0], [2, 2, 0], [3, 3, 3]])
    )
    # Columns only:
    C = M.scale_rows_and_columns(col_scale=col_scale)
    assert_allclose(
        C.materialize(n), jnp.array([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
    )
    # Both rows and columns:
    B = M.scale_rows_and_columns(row_scale=row_scale, col_scale=col_scale)
    assert_allclose(
        B.materialize(n), jnp.array([[1, 0, 0], [2, 4, 0], [3, 6, 9]])
    )

  @parameterized.named_parameters(matrix_and_n_params())
  def test_scale_rows_and_columns_vs_materialized(self, A, n):
    row_scales = jnp.arange(1, n + 1)
    col_scales = jnp.linspace(1.0, 0.1, num=n)
    A = A(n)
    assert_allclose(
        A.scale_rows_and_columns(
            row_scale=row_scales, col_scale=col_scales
        ).materialize(n),
        jnp.diag(row_scales) @ A.materialize(n) @ jnp.diag(col_scales),
    )

  @parameterized.named_parameters(matrix_and_n_params())
  def test_multiply_array_on_tensor(self, A, n):
    A = A(n)
    Z = jnp.arange(n * 2 * 3).reshape((n, 2, 3))
    assert_allclose(A @ Z, jnp.tensordot(A.materialize(n), Z, axes=1))


if __name__ == '__main__':
  absltest.main()
