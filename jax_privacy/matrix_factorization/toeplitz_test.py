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

# Note: When building from source, these tests should use jax with compiler
# optimizations, e.g. `-c opt`. Otherwise runtimes may be excessive and timeouts
# may occur. If `jax` is installed via a pip package, it should already be
# optimized. Another likely cause of timeouts is that the default hypothesis
# profile runs too many test inputs (`num_examples`), so consider reducing the
# number of examples it generates via `HYPOTHESIS_PROFILE=dpftrl_default` (see
# test_utils.py).

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import hypothesis
from hypothesis import strategies as st
import jax
from jax import numpy as jnp
from jax_privacy.matrix_factorization import dense
from jax_privacy.matrix_factorization import sensitivity
from jax_privacy.matrix_factorization import test_utils
from jax_privacy.matrix_factorization import toeplitz
import numpy as np

# Note: Some of these tests fail quite badly without this line:
jax.config.update('jax_enable_x64', True)

test_utils.configure_hypothesis()

# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


# Tuples of (coef, n) to test.
NAMED_C_MATRIX_PARAMS = [
    ('n=1', [7.3], 3),
    ('n=len(coef)', [1, 0.5, 0.2], 3),
    ('n>len(coef)', [1, 0.7, 2.0], 4),
    ('neg_below_main_diag', [1, 0.0, -0.2], 5),
    ('n=25', np.linspace(1, 0.001, num=25), 25),
    ('n=None', np.linspace(0.5, 2.0, num=11), None),
]

# Tuples of coef to test.
NAMED_C_INV_MATRIX_PARAMS = [
    ('n=1', [7.3]),
    ('n=3', [1, -0.5, -0.2]),
    ('n=5', [5.0, -0.7, -2.0, -1.0, -0.3]),
    ('pos_below_main_diag', [1, 0.0, 0.7]),
    ('n=16', np.concatenate([np.ones(1), np.linspace(-0.5, -0.001, num=15)])),
]

# List of Toeplitz coefficients for use in tests
COEFS = [
    [0.1],
    [1.0],
    [0.1, 0.5, 1.0],  # Increasing
    np.linspace(1.0, 0.1, num=16),
    [1.0, 0.0, 0.0, -0.5, 0.0, 0.3],
]


class ToeplitzTest(parameterized.TestCase):

  def test_reconcile(self):
    # We test the `_reconcile` function here, so in general we only need
    # to test the case where n is not None in tests below.

    def tuple_equal(result, expected):
      jax.tree_util.tree_map(np.testing.assert_allclose, result, expected)

    # n is set
    tuple_equal(toeplitz._reconcile([1], 1), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2], 1), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2], 2), (jnp.array([1, 2]), 2))
    tuple_equal(toeplitz._reconcile([1, 2], 3), (jnp.array([1, 2]), 3))

    # n is None
    tuple_equal(toeplitz._reconcile([1]), (jnp.ones(1), 1))
    tuple_equal(toeplitz._reconcile([1, 2]), (jnp.array([1, 2]), 2))
    tuple_equal(toeplitz._reconcile([1, 2, 3]), (jnp.array([1, 2, 3]), 3))

  def test_materialize_lower_triangular(self):
    np.testing.assert_allclose(
        jnp.array([[1]]), toeplitz.materialize_lower_triangular(jnp.array([1]))
    )
    np.testing.assert_allclose(
        # pyformat: disable
        jnp.array([[2, 0, 0],
                   [1, 2, 0],
                   [3, 1, 2]]),
        # pyformat: enable
        toeplitz.materialize_lower_triangular(jnp.array([2, 1, 3])),
    )

  def test_materialize_lower_triangular_n(self):
    # n < len(coef)
    np.testing.assert_allclose(
        jnp.array([[1]]),
        toeplitz.materialize_lower_triangular(coef=[1, 2, 3], n=1),
    )

    # n > len(coef)
    np.testing.assert_allclose(
        # pyformat: disable
        jnp.array([[2, 0, 0],
                   [1, 2, 0],
                   [0, 1, 2]]),
        # pyformat: enable
        toeplitz.materialize_lower_triangular(jnp.array([2, 1]), n=3),
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 10, 100])
  def test_fhu_coefs(self, n):
    coef = toeplitz.optimal_max_error_strategy_coefs(n)
    C = toeplitz.materialize_lower_triangular(coef)
    np.testing.assert_allclose(
        np.tri(n),
        C @ C,
        atol=1e-6,
    )

  @parameterized.named_parameters((f'{n=}', n) for n in [1, 2, 10, 100])
  def test_fhu_inv_coefs(self, n):
    coef = toeplitz.optimal_max_error_strategy_coefs(n)
    C = toeplitz.materialize_lower_triangular(coef)
    inv_coef = toeplitz.optimal_max_error_noising_coefs(n)
    C_inv = toeplitz.materialize_lower_triangular(inv_coef)
    np.testing.assert_allclose(
        np.eye(n),
        C @ C_inv,
        atol=1e-6,
    )

  @parameterized.named_parameters(
      ('full', 16, 16),
      ('basic', 15, 6),
      ('single_band', 16, 1),
      ('single_iter', 1, 1),
  )
  def test_solve_toeplitz(self, n, bands):
    # We set the first coefficient to 1.0 to make sure the matrix is
    # reasonably well conditioned.
    coef = jax.random.normal(jax.random.key(1234), (bands,)).at[0].set(1.0)
    b = jax.random.normal(jax.random.key(5678), (n,))
    x = toeplitz.solve_banded(coef, b)

    C = toeplitz.materialize_lower_triangular(
        jnp.zeros(n).at[: len(coef)].set(coef)
    )
    np.testing.assert_allclose(x, np.linalg.solve(C, b), atol=1e-6)

  @hypothesis.given(
      lhs_coef=st.sampled_from(COEFS),
      rhs_coef=st.sampled_from(COEFS),
      n=st.integers(1, 32),
  )
  def test_multiply(self, lhs_coef, rhs_coef, n):
    L = toeplitz.materialize_lower_triangular(lhs_coef, n)
    R = toeplitz.materialize_lower_triangular(rhs_coef, n)
    expected_coef = (L @ R)[:, 0]
    actual_coef = toeplitz.multiply(lhs_coef, rhs_coef, n)
    actual_coef = toeplitz.pad_coefs_to_n(actual_coef, n)
    np.testing.assert_allclose(actual_coef, expected_coef, atol=1e-6)

  def test_multiply_no_n(self):
    lhs_coef = jnp.array([1, 2, 3, 4])
    rhs_coef = jnp.array([5, 6, 0, 0])
    L = toeplitz.materialize_lower_triangular(lhs_coef)
    R = toeplitz.materialize_lower_triangular(rhs_coef)
    expected_coef = (L @ R)[:, 0]
    actual_coef = toeplitz.multiply(lhs_coef, rhs_coef, n=None)
    np.testing.assert_allclose(actual_coef, expected_coef, atol=1e-6)

  def test_multiply_checks(self):
    lhs_coef = jnp.array([1, 2, 3, 4])
    rhs_coef = jnp.array([5, 6])
    with self.assertRaisesRegex(
        ValueError, 'lhs_coef and rhs_coef must have the same length'
    ):
      toeplitz.multiply(lhs_coef, rhs_coef, n=None)

  @hypothesis.given(n=st.integers(1, 32))
  def test_inverse_is_column_normalized(self, n: int):
    coef = jnp.array([1, 0.8, 0.6, 0.4])
    coef = coef / jnp.linalg.norm(coef[:n])
    C_inv = toeplitz.inverse_as_streaming_matrix(coef, column_normalize_for_n=n)
    C = jnp.linalg.inv(C_inv.materialize(n))
    np.testing.assert_allclose(
        jnp.linalg.norm(C, axis=0),
        jnp.ones(n),
        atol=1e-6,
    )

  def test_column_normalized_for_n_raises_on_bad_coef(self):
    coef = jnp.array([1, 0.8, 0.6, 0.4])
    with self.assertRaisesRegex(ValueError, 'must have an L2 norm of 1.0'):
      toeplitz.inverse_as_streaming_matrix(coef, column_normalize_for_n=10)

  @hypothesis.given(n=st.integers(1, 32))
  def test_inverse_is_correct(self, n: int):
    coef = jnp.array([1, 0.8, 0.6, 0.4])
    C_inv = toeplitz.inverse_as_streaming_matrix(coef, None)
    np.testing.assert_allclose(
        jnp.linalg.inv(C_inv.materialize(n)),
        toeplitz.materialize_lower_triangular(coef, n),
        atol=1e-6,
    )

  @parameterized.named_parameters(*[(f'{n=}', n) for n in [1, 2, 5, 10]])
  def test_inverse_toeplitz_coef(self, n):
    coef = jnp.linspace(1.0, 0.0, num=n)
    C = toeplitz.materialize_lower_triangular(coef)
    inv_coef = toeplitz.inverse_coef(coef)
    Cinv = toeplitz.materialize_lower_triangular(inv_coef)
    np.testing.assert_allclose(C @ Cinv, jnp.eye(n), atol=1e-6)

  @parameterized.named_parameters(*[(f'{b=}', b) for b in [1, 2, 5, 10]])
  def test_banded_inverse_toeplitz_coef(self, b):
    n = 2 * b
    coef = jnp.linspace(1.0, 0.0, num=b)
    C = toeplitz.materialize_lower_triangular(coef, n)
    inv_coef = toeplitz.inverse_coef(coef, n)
    Cinv = toeplitz.materialize_lower_triangular(inv_coef, n)
    np.testing.assert_allclose(C @ Cinv, jnp.eye(n), atol=1e-6)

  @parameterized.product(n=[1, 2, 11], coef_idx=[0, 1, 2])
  def test_single_participation_sensitivity(self, n, coef_idx):
    coefs = [[1.0], [1.0, 0.5], jnp.linspace(1.0, 0.1, num=11)]
    coef = coefs[coef_idx]
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, n=n, min_sep=1, max_participations=1
        ),
        toeplitz.sensitivity_squared(coef, n),
    )

  @parameterized.product(n=[1, 2, 3, 5, 11], k=[1, 4, 11])
  def test_sensitivity_squared_identity(self, n, k):
    actual_max_participations = min(n, k)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            strategy_coef=[1.0], n=n, min_sep=1, max_participations=k
        ),
        actual_max_participations,
    )

  @parameterized.product(n=[1, 2, 3, 5, 11], k=[1, 4, 11])
  def test_sensitivity_squared_tri(self, n, k):
    C = np.tri(n)
    coef = np.ones(n)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, n=n, min_sep=1, max_participations=k
        ),
        # sensitivity**2 is the sum of the L2-norm-squared of the
        # first k columns of C:
        np.sum(np.sum(C[:, :k], axis=1) ** 2),
    )

  @parameterized.product(
      n_and_epochs=[
          (1, 1),
          (2, 1),
          (2, 2),
          (10, 1),
          (10, 5),
          (10, 10),
      ],
      bands=[2, 13, 10, 11],
  )
  def test_sensitivity_squared_fixed_epoch(self, n_and_epochs, bands):
    n, epochs = n_and_epochs
    # For a Toeplitz matrix with positive decreasing entries, the worst
    # case is a fixed-epoch participation pattern. Note we must have
    # n % epochs == 0 here.
    coef = jnp.linspace(1.0, 0.001, num=bands)
    C = toeplitz.materialize_lower_triangular(coef, n=n)
    expected = sensitivity.fixed_epoch_sensitivity(C, epochs) ** 2
    min_sep = n // epochs

    # We should get the same result if we set a higher more max_participations,
    # or infer it from n and min_sep:
    for max_particpations in [epochs, epochs + 10, None]:
      np.testing.assert_allclose(
          toeplitz.minsep_sensitivity_squared(
              coef,
              min_sep=min_sep,
              max_participations=max_particpations,
              n=n,
          ),
          expected,
          atol=1e-6,
      )

  @parameterized.product(
      n=[1, 5, 50],
      max_participations=[1, 2, 5, 50],
  )
  def test_sensitivity_squared_min_sep_one(self, n, max_participations):
    coef = jnp.sqrt(jnp.linspace(1.0, 0.001, num=n))
    C = toeplitz.materialize_lower_triangular(coef)
    np.testing.assert_allclose(
        toeplitz.minsep_sensitivity_squared(
            coef, min_sep=1, max_participations=max_participations
        ),
        np.linalg.norm(np.sum(C[:, :max_participations], axis=1)) ** 2,
        rtol=1e-10,
    )

  def test_sensitivity_squared_bad_input(self):
    with self.assertRaisesRegex(ValueError, 'coef must be non-negative'):
      toeplitz.minsep_sensitivity_squared(strategy_coef=[-1.0], min_sep=1)
    with self.assertRaisesRegex(ValueError, 'coef must be non-increasing'):
      toeplitz.minsep_sensitivity_squared(
          strategy_coef=[1.0, 2.0, 1.0], min_sep=1
      )
    with self.assertRaisesRegex(ValueError, 'min_sep must be positive'):
      toeplitz.minsep_sensitivity_squared(strategy_coef=[1.0], min_sep=0)


def _sensitivity_squared(coef: jnp.ndarray, n: int | None) -> jnp.ndarray:
  """Single-participation sensitivity."""
  coef, _ = toeplitz._reconcile(coef, n)
  return toeplitz._l2_norm_squared(coef)


# TODO: b/329444015 - Replace these helpers with calls do dense.py
# when error computations are available there.


def _per_query_error(
    coef: jnp.ndarray, n: int | None, workload_coef: jax.Array | None = None
) -> jax.Array:
  coef, n = toeplitz._reconcile(coef, n)
  if workload_coef is None:
    workload_coef = jnp.ones(n)
  workload_coef = toeplitz.pad_coefs_to_n(workload_coef, n)
  C = toeplitz.materialize_lower_triangular(coef, n=n)
  A = toeplitz.materialize_lower_triangular(workload_coef, n=n)
  return dense.per_query_error(strategy_matrix=C, workload_matrix=A)


def _mean_error(coef: jnp.ndarray, n: int | None) -> jnp.ndarray:
  coef, n = toeplitz._reconcile(coef, n)
  C = toeplitz.materialize_lower_triangular(coef, n=n)
  return dense.mean_error(strategy_matrix=C)


def _mean_error_for_inv(c_inv_coef: jnp.ndarray) -> jnp.ndarray:
  c_inv_coef, n = toeplitz._reconcile(c_inv_coef, None)
  C_inv = toeplitz.materialize_lower_triangular(c_inv_coef, n=n)
  return dense.mean_error(noising_matrix=C_inv)


def _max_error(coef: jnp.ndarray, n: int | None) -> jnp.ndarray:
  coef, n = toeplitz._reconcile(coef, n)
  C = toeplitz.materialize_lower_triangular(coef, n=n)
  return dense.max_error(strategy_matrix=C)


def _max_error_for_inv(c_inv_coef: jnp.ndarray) -> jnp.ndarray:
  c_inv_coef, n = toeplitz._reconcile(c_inv_coef, None)
  C_inv = toeplitz.materialize_lower_triangular(c_inv_coef, n=n)
  return dense.max_error(noising_matrix=C_inv)


class ToeplitzErrorTest(parameterized.TestCase):

  @hypothesis.given(n=st.integers(1, 14))
  def test_mean_error_identity(self, n):
    expected = _mean_error(coef=[1], n=n)
    # We can also compute this directly as n(n+1) / 2 / n
    np.testing.assert_allclose(expected, (n + 1) / 2)
    # Test both implicit and explicit coefs
    coef = jnp.zeros(n).at[0].set(1.0)
    np.testing.assert_allclose(
        toeplitz.mean_error(strategy_coef=coef), expected
    )
    np.testing.assert_allclose(
        toeplitz.mean_error(strategy_coef=[1], n=n), expected
    )

    # The inverse coefficients are the same for the identity matrix:
    np.testing.assert_allclose(toeplitz.mean_error(noising_coef=coef), expected)

  @hypothesis.given(n=st.integers(1, 14))
  def test_max_error_identity(self, n):
    expected = _max_error([1], n=n)
    # We can also compute this directly as n
    np.testing.assert_allclose(expected, n)

    # Test both implicit and explicit coefs
    coef = jnp.zeros(n).at[0].set(1.0)
    np.testing.assert_allclose(toeplitz.max_error(strategy_coef=coef), expected)
    np.testing.assert_allclose(
        toeplitz.max_error(strategy_coef=[1], n=n), expected
    )

    # The inverse coefficients are the same for the identity matrix:
    np.testing.assert_allclose(toeplitz.max_error(noising_coef=coef), expected)

  @hypothesis.given(
      name_coef_n_tuple=st.sampled_from(NAMED_C_MATRIX_PARAMS),
      workload=st.sampled_from(
          ['default', 'prefix_sum', 'eye', 'banded', 'extra_entries']
      ),
      config=st.sampled_from(['jit', 'skip_checks=True', 'skip_checks=False']),
  )
  @hypothesis.settings(max_examples=test_utils.scale_max_examples(10))
  def test_per_query_error(self, name_coef_n_tuple, workload, config):
    _, coef, n = name_coef_n_tuple
    _, true_n = toeplitz._reconcile(coef, n)
    if workload == 'default':
      workload_coef = None
    elif workload == 'prefix_sum':
      workload_coef = jnp.ones(true_n)
    elif workload == 'eye':
      workload_coef = jnp.ones(1)  # Rest implicitly 0
    elif workload == 'banded':
      workload_coef = jnp.array([1, 0.9, 0.5, 0.1])
    elif workload == 'extra_entries':
      workload_coef = jnp.linspace(0.99, 0.1, num=true_n + 5)
    else:
      raise ValueError(f'Unknown workload: {workload}')

    per_query_error = toeplitz.per_query_error
    if config == 'jit':
      per_query_error = jax.jit(
          per_query_error, static_argnames=['skip_checks', 'n']
      )
      skip_checks = True
    elif config == 'skip_checks=True':
      skip_checks = True
    elif config == 'skip_checks=False':
      skip_checks = False
    else:
      raise ValueError(f'Unknown config={config}')

    inv_coef = toeplitz.inverse_coef(coef, n)
    expected = _per_query_error(coef, n, workload_coef=workload_coef)
    kwargs = {
        'skip_checks': skip_checks,
        'workload_coef': workload_coef,
        'n': n,
    }
    np.testing.assert_allclose(
        per_query_error(strategy_coef=coef, **kwargs),
        expected,
        err_msg='Failure computing error for strategy_coef',
    )
    np.testing.assert_allclose(
        per_query_error(noising_coef=inv_coef, **kwargs),
        expected,
        err_msg='Failure computing error for noising_coef',
    )

  @parameterized.named_parameters(NAMED_C_MATRIX_PARAMS)
  def test_mean_error(self, coef, n):
    expected = _mean_error(coef, n)
    np.testing.assert_allclose(
        toeplitz.mean_error(strategy_coef=coef, n=n), expected
    )

  @parameterized.named_parameters(NAMED_C_MATRIX_PARAMS)
  def test_max_error(self, coef, n):
    expected = _max_error(coef, n)
    np.testing.assert_allclose(
        toeplitz.max_error(strategy_coef=coef, n=n), expected
    )

  @parameterized.named_parameters(NAMED_C_INV_MATRIX_PARAMS)
  def test_mean_error_for_inv(self, c_inv_coef):
    expected = _mean_error_for_inv(c_inv_coef)
    np.testing.assert_allclose(
        toeplitz.mean_error(noising_coef=c_inv_coef), expected
    )

  @parameterized.named_parameters(NAMED_C_INV_MATRIX_PARAMS)
  def test_max_error_for_inv(self, c_inv_coef):
    expected = _max_error_for_inv(c_inv_coef)
    np.testing.assert_allclose(
        toeplitz.max_error(noising_coef=c_inv_coef), expected
    )

  @hypothesis.settings(deadline=None, max_examples=10)
  @hypothesis.given(
      n=st.integers(1, 8),
  )
  def test_mean_loss(self, n):
    coef = jnp.array([1.0, 0.4, 0.3, 0.1])
    loss = toeplitz.mean_loss(coef, n)

    expected = _sensitivity_squared(coef, n) * _mean_error(coef, n)
    np.testing.assert_allclose(loss, expected)

    # Confirm loss is scale invariant
    np.testing.assert_allclose(loss, toeplitz.mean_loss(2.3 * coef, n))


class ToeplitzOptimizationTest(parameterized.TestCase):

  @parameterized.named_parameters((f'{b}_bands', b) for b in [1, 5, 10, 15])
  def test_toeplitz_more_bands_is_better(self, b):
    coef1 = toeplitz.optimize_banded_toeplitz(16, b, max_optimizer_steps=25)
    coef2 = toeplitz.optimize_banded_toeplitz(16, b + 1, max_optimizer_steps=25)
    self.assertLessEqual(
        toeplitz.mean_loss(coef2, 16), toeplitz.mean_loss(coef1, 16)
    )


class ToeplitzAmplificationTest(parameterized.TestCase):

  def test_optimize_banded_toeplitz_for_amplifications(self):
    dataset_size = 1024
    batch_size = 16
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=2.0,
        delta=1e-6,
    )
    bands = len(coef)
    self.assertGreater(stddev, 0)
    self.assertLessEqual(bands, dataset_size // batch_size)

  def test_large_epsilon(self):
    # Note: This step requires multiple steps of optimization
    # and noise calibration, so it can take a while.

    # With a large epsilon, correlated noise is more effective than
    # amplification, so we expect the maximum number of bands is optimal.
    dataset_size = 1000
    batch_size = 100
    max_bands = dataset_size // batch_size
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=20.0,
        delta=1e-6,
    )
    self.assertGreater(stddev, 0)
    bands = len(coef)
    self.assertEqual(bands, max_bands)

  def test_small_epsilon(self):
    # Note: This step requires multiple steps of optimization
    # and noise calibration, so it can take a while.

    # With a small epsilon, amplification is more effective than
    # correlated noise, so we expect a single band is optimal.
    dataset_size = 1000
    batch_size = 100
    coef, stddev = toeplitz.optimize_coefs_for_amplifications(
        n=100,
        dataset_size=dataset_size,
        expected_batch_size=batch_size,
        epsilon=0.1,
        delta=1e-6,
    )
    self.assertGreater(stddev, 0)
    bands = len(coef)
    self.assertEqual(bands, 1)

  def test_helper(self):
    n = 100
    dataset_size = 100
    batch_size = 10
    epsilon = 2.0
    delta = 1e-6
    helper = toeplitz._AmplifiedBandMFHelper(
        n=n,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epsilon=epsilon,
        delta=delta,
    )
    max_bands = 10
    self.assertEqual(helper.max_bands, max_bands)
    self.assertEqual(helper.sensitivity(coef=jnp.array([1.0])), 1.0)
    self.assertEqual(
        helper.sensitivity(coef=jnp.array([1.0, 1.0])), jnp.sqrt(2)
    )

  def test_helper_stddev_at_max_bands(self):
    n = 100
    dataset_size = 100
    batch_size = 10
    epsilon = 2.0
    delta = 1e-6
    helper = toeplitz._AmplifiedBandMFHelper(
        n=n,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epsilon=epsilon,
        delta=delta,
    )
    max_bands = 10
    assert max_bands == dataset_size // batch_size
    # If bands == max_bands, then there is no amplification, and
    # we should get the same standard deviation as the non-amplified case
    # assuming bands-min-sep participation.
    k = n // max_bands
    assert k * max_bands == n
    coef = toeplitz.optimal_max_error_strategy_coefs(max_bands)

    b_min_sep_sens = jnp.sqrt(
        toeplitz.minsep_sensitivity_squared(
            coef, min_sep=max_bands, max_participations=k, n=n
        )
    )

    total_nm = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
        make_event_from_param=lambda nm: dp_accounting.GaussianDpEvent(
            noise_multiplier=nm
        ),
        target_epsilon=epsilon,
        target_delta=delta,
        bracket_interval=dp_accounting.ExplicitBracketInterval(0.01, 10.0),
    )
    expected_stddev = total_nm * b_min_sep_sens

    helper_stddev = helper.required_stddev(coef=coef)
    np.testing.assert_allclose(
        helper_stddev, expected_stddev, rtol=1e-5, atol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
