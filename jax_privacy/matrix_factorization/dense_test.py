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

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
from jax_privacy.matrix_factorization import dense
from jax_privacy.matrix_factorization import test_utils
import numpy as np

test_utils.configure_hypothesis()


# Disabling pylint so we can use single-letter capitals to denote matrices
# as per notation table in README.md
# pylint:disable=invalid-name


class StoreStateCallback:

  def __init__(self):
    self.step_info = None

  def __call__(self, step_info):
    self.step_info = step_info


@dataclasses.dataclass(frozen=True)
class ErrorTestCase:
  A: np.ndarray
  C: np.ndarray
  C_inv: np.ndarray
  expected_per_query_error: np.ndarray
  expected_max_error: np.ndarray
  expected_mean_error: np.ndarray


def diagonal_error_test_cases():
  """Returns a list of ErrorTestCases for diagonal matrices."""
  test_cases = []
  for n in [1, 2, 11]:
    A = np.eye(n)
    bd = np.linspace(1.5, 4.0 / n, num=n)
    C = np.diag(1 / bd)
    C_inv = np.diag(bd)
    expected_per_query_error = bd**2
    test_cases.append(
        ErrorTestCase(
            A=A,
            C=C,
            C_inv=C_inv,
            expected_per_query_error=bd**2,
            expected_max_error=np.max(expected_per_query_error),
            expected_mean_error=np.mean(expected_per_query_error),
        )
    )
  return test_cases


ERROR_TEST_CASES = diagonal_error_test_cases() + [
    # 2x2 prefix sum, simple factorization
    ErrorTestCase(
        A=np.tri(2),
        C=np.array([[1, 0], [0.5, 1]]),
        C_inv=np.array([[1, 0], [-0.5, 1]]),
        # B = [[1, 0], [0.5, 1]]
        expected_per_query_error=np.array([1, 1.25]),
        expected_max_error=np.array(1.25),
        expected_mean_error=np.array((1 + 1.25) / 2),
    ),
    # A non-square example (n=2 tree aggregation):
    ErrorTestCase(
        A=np.tri(2),
        C=np.array([[1, 0], [0, 1], [1, 1]]),
        # "Full Honaker C_inv (also, the Moore-Penrose pseudoinverse)"
        C_inv=np.array([[2 / 3, -1 / 3, 1 / 3], [-1 / 3, 2 / 3, 1 / 3]]),
        expected_per_query_error=np.array([2 / 3, 2 / 3]),
        expected_max_error=np.array(2 / 3),
        expected_mean_error=np.array(2 / 3),
    ),
]


class DenseTest(parameterized.TestCase):

  def check_symmetric(self, X):
    return np.testing.assert_allclose(X, X.T, atol=1e-8)

  def check_psd(self, X):
    """Checks if a given matrix is PSD."""
    return self.assertGreater(np.linalg.eigvals(X).min(), 0)

  def check_banded(self, X, bands):
    """Checks if a given matrix is banded (see README.md for definition)."""
    n = X.shape[0]
    for i in range(n):
      for j in range(n):
        if abs(i - j) >= bands:
          self.assertEqual(X[i, j], 0)

  def check_equal_norm(self, C):
    """Checks that column norms of matrix are equal."""
    norms = np.linalg.norm(C, axis=0)
    self.assertAlmostEqual(norms.max(), norms.min())

  def check_lower_loss_than_identity(self, A, X, epochs):
    init = np.eye(X.shape[0]) / epochs
    self.assertLessEqual(
        dense._mean_loss_and_gradient(X, A)[0],
        dense._mean_loss_and_gradient(init, A)[0],
    )

  @hypothesis.given(
      test_case=st.sampled_from(ERROR_TEST_CASES),
      config=st.sampled_from(['jit', 'skip_checks=True', 'skip_checks=False']),
  )
  @hypothesis.settings(max_examples=test_utils.scale_max_examples(5))
  def test_error(self, test_case, config):

    if config == 'jit':
      maybe_jit = functools.partial(jax.jit, static_argnames=['skip_checks'])
      skip_checks = True
    elif config == 'skip_checks=True':
      maybe_jit = lambda x: x
      skip_checks = True
    elif config == 'skip_checks=False':
      maybe_jit = lambda x: x
      skip_checks = False
    else:
      raise ValueError(f'Unknown config={config}')

    kwargs = {'workload_matrix': test_case.A, 'skip_checks': skip_checks}

    # We could probably use smaller tolerances if we
    # enabled x64.
    assert_allclose = functools.partial(
        np.testing.assert_allclose, rtol=1e-6, atol=1e-6
    )

    per_query_error = maybe_jit(dense.per_query_error)
    max_error = maybe_jit(dense.max_error)
    mean_error = maybe_jit(dense.mean_error)

    # Test errors for C (if C is square):
    if test_case.C.shape[0] == test_case.C.shape[1]:
      assert_allclose(
          per_query_error(strategy_matrix=test_case.C, **kwargs),
          test_case.expected_per_query_error,
          err_msg='Failure computing per-query error for strategy_matrix',
      )
      assert_allclose(
          max_error(strategy_matrix=test_case.C, **kwargs),
          test_case.expected_max_error,
          err_msg='Failure computing max error for strategy_matrix',
      )
      assert_allclose(
          mean_error(strategy_matrix=test_case.C, **kwargs),
          test_case.expected_mean_error,
          err_msg='Failure computing mean error for strategy_matrix',
      )

    # Test errors from C^{-1}
    assert_allclose(
        per_query_error(noising_matrix=test_case.C_inv, **kwargs),
        test_case.expected_per_query_error,
        err_msg='Failure computing per_query error for noising_matrix',
    )
    assert_allclose(
        max_error(noising_matrix=test_case.C_inv, **kwargs),
        test_case.expected_max_error,
        err_msg='Failure computing max error for noising_matrix',
    )
    assert_allclose(
        mean_error(noising_matrix=test_case.C_inv, **kwargs),
        test_case.expected_mean_error,
        err_msg='Failure computing mean error for noising_matrix',
    )

  # pylint:disable=bad-whitespace
  # pyformat:disable
  @parameterized.named_parameters(
      ('single_epoch', 1,  None, False),
      ('full_batch',   16, None, False),
      ('multi_epoch',  4,  None, False),
      ('banded',       4,  4,    False),
      ('equal_norm',   4,  None, True))
  # pyformat:enable
  # pylint:enable=bad-whitespace
  def test_optimization_worked(self, epochs, bands, equal_norm):
    A = np.tri(16)
    callback = StoreStateCallback()
    C = dense.optimize(
        16,
        epochs=epochs,
        bands=bands,
        equal_norm=equal_norm,
        A=A,
        max_optimizer_steps=250,
        callback=callback,
    )
    X = C.T @ C
    self.check_symmetric(X)
    self.check_psd(X)
    self.check_lower_loss_than_identity(A, X, epochs)
    grad_max = np.abs(
        callback.step_info.grad  # pytype: disable=attribute-error
    ).max()
    self.assertLessEqual(grad_max, 1e-3)

  def test_identity(self):
    n = 16
    A = np.eye(n)
    C = dense.optimize(n, epochs=1, max_optimizer_steps=100, A=A)
    np.testing.assert_allclose(C, A)

  def test_callback_called_every_step(self):
    num_times_called = 0

    def callback(_):
      nonlocal num_times_called
      num_times_called += 1

    n = 16
    A = np.eye(n)
    dense.optimize(n, epochs=1, A=A, max_optimizer_steps=100, callback=callback)
    self.assertEqual(num_times_called, 100)

  def test_single_band(self):
    n = 16
    C = dense.optimize(n, epochs=1, bands=1, max_optimizer_steps=100)
    np.testing.assert_allclose(C, np.eye(n))

    C = dense.optimize(n, epochs=n, bands=1, max_optimizer_steps=100)
    self.assertAlmostEqual(np.diag(C.T @ C).sum(), 1.0)
    self.check_banded(C, 1)

  def test_colnorm(self):
    n = 16
    C = dense.optimize(n, epochs=4, equal_norm=True)
    self.check_equal_norm(C)

  def test_finite_differences(self):
    with jax.experimental.enable_x64():
      n = 16
      A = np.tri(n)
      X = np.eye(n)
      _, dX = dense._mean_loss_and_gradient(X, A)
      dX = dX.at[np.diag_indices(n)].set(0)
      jax.block_until_ready(dX)
      finite_diff_approx = np.zeros((n, n))
      diff = 1e-6
      for i in range(n):
        for j in range(i):
          X[i, j] = X[j, i] = diff
          L1 = dense._mean_loss_and_gradient(X, A)[0]
          jax.block_until_ready(L1)
          X[i, j] = X[j, i] = -diff
          L2 = dense._mean_loss_and_gradient(X, A)[0]
          jax.block_until_ready(L2)
          X[i, j] = X[j, i] = 0
          finite_diff_approx[i, j] = finite_diff_approx[j, i] = (L1 - L2) / (
              4 * diff
          )
      np.testing.assert_allclose(dX, finite_diff_approx, rtol=1e-4)


# pylint:enable=invalid-name
if __name__ == '__main__':
  absltest.main()
