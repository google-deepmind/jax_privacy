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

"""Tests for workloads module."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_privacy.experimental import workloads
import numpy as np


def _assert_gram_equals_dense_gram(w):
  """Checks that w.gram == w.dense.T @ w.dense."""
  expected = w.dense.T @ w.dense
  np.testing.assert_allclose(w.gram, expected, atol=1e-5)


class PrefixSumTest(absltest.TestCase):

  def test_dense_is_lower_triangular_ones(self):
    w = workloads.PrefixSum(n=5)
    np.testing.assert_array_equal(w.dense, jnp.tri(5))

  def test_n(self):
    self.assertEqual(workloads.PrefixSum(n=7).n, 7)

  def test_dense_shape(self):
    self.assertEqual(workloads.PrefixSum(n=10).dense.shape, (10, 10))

  def test_streaming_matrix_matches_dense(self):
    w = workloads.PrefixSum(n=8)
    np.testing.assert_allclose(
        w.streaming_matrix.materialize(8), w.dense, atol=1e-6
    )

  def test_toeplitz_coefs(self):
    np.testing.assert_array_equal(
        workloads.PrefixSum(n=5).toeplitz_coefs, jnp.ones(5)
    )

  def test_gram(self):
    _assert_gram_equals_dense_gram(workloads.PrefixSum(n=4))


class SuffixSumTest(absltest.TestCase):

  def test_dense_is_reversed_prefix(self):
    w = workloads.SuffixSum(n=5)
    np.testing.assert_array_equal(w.dense, jnp.tri(5)[::-1, ::-1])

  def test_dense_shape(self):
    self.assertEqual(workloads.SuffixSum(n=6).dense.shape, (6, 6))

  def test_streaming_matrix_is_none(self):
    self.assertIsNone(workloads.SuffixSum(n=5).streaming_matrix)

  def test_toeplitz_coefs_is_none(self):
    self.assertIsNone(workloads.SuffixSum(n=5).toeplitz_coefs)


class IdentityTest(absltest.TestCase):

  def test_dense_is_eye(self):
    np.testing.assert_array_equal(workloads.Identity(n=5).dense, jnp.eye(5))

  def test_streaming_matrix_matches_dense(self):
    w = workloads.Identity(n=6)
    np.testing.assert_allclose(
        w.streaming_matrix.materialize(6), w.dense, atol=1e-6
    )

  def test_toeplitz_coefs(self):
    expected = jnp.zeros(5).at[0].set(1.0)
    np.testing.assert_array_equal(
        workloads.Identity(n=5).toeplitz_coefs, expected
    )

  def test_gram_is_identity(self):
    np.testing.assert_allclose(
        workloads.Identity(n=4).gram, jnp.eye(4), atol=1e-6
    )


class AllRangeTest(absltest.TestCase):

  def test_shape(self):
    n = 4
    w = workloads.AllRange(n=n)
    self.assertEqual(w.dense.shape, (n * (n + 1) // 2, n))

  def test_contains_identity_rows(self):
    w = workloads.AllRange(n=4)
    dense = w.dense
    for i in range(4):
      row = jnp.zeros(4).at[i].set(1.0)
      self.assertTrue(
          any(jnp.allclose(dense[j], row) for j in range(dense.shape[0])),
          f'Missing single-element range [{i}]',
      )

  def test_contains_full_range(self):
    dense = workloads.AllRange(n=4).dense
    self.assertTrue(
        any(jnp.allclose(dense[j], jnp.ones(4)) for j in range(dense.shape[0]))
    )

  def test_gram(self):
    _assert_gram_equals_dense_gram(workloads.AllRange(n=4))


class DenseWorkloadTest(absltest.TestCase):

  def test_dense_returns_matrix(self):
    m = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(workloads.DenseWorkload(matrix=m).dense, m)

  def test_n_inferred_from_matrix(self):
    self.assertEqual(workloads.DenseWorkload(matrix=jnp.ones((3, 7))).n, 7)

  def test_gram(self):
    m = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    _assert_gram_equals_dense_gram(workloads.DenseWorkload(matrix=m))


class MomentumCooldownTest(absltest.TestCase):

  def test_dense_shape(self):
    lr = jnp.ones(10)
    w = workloads.MomentumCooldown(momentum=0.9, learning_rates=lr, n=10)
    self.assertEqual(w.dense.shape, (10, 10))

  def test_streaming_matrix_matches_dense(self):
    lr = jnp.ones(8)
    w = workloads.MomentumCooldown(momentum=0.5, learning_rates=lr, n=8)
    np.testing.assert_allclose(
        w.streaming_matrix.materialize(8), w.dense, atol=1e-5
    )

  def test_zero_momentum_is_lower_triangular(self):
    lr = jnp.array([1.0, 2.0, 3.0])
    w = workloads.MomentumCooldown(momentum=0.0, learning_rates=lr, n=3)
    self.assertTrue(jnp.allclose(w.dense, jnp.tril(w.dense)))


class ScalarScaledTest(absltest.TestCase):

  def test_dense(self):
    w = workloads.ScalarScaled(alpha=2.0, base=workloads.PrefixSum(n=4))
    np.testing.assert_allclose(w.dense, 2.0 * jnp.tri(4), atol=1e-6)

  def test_n_from_base(self):
    w = workloads.ScalarScaled(alpha=3.0, base=workloads.Identity(n=5))
    self.assertEqual(w.n, 5)

  def test_toeplitz_coefs_propagates(self):
    w = workloads.ScalarScaled(alpha=0.5, base=workloads.PrefixSum(n=4))
    np.testing.assert_allclose(w.toeplitz_coefs, 0.5 * jnp.ones(4), atol=1e-6)

  def test_toeplitz_coefs_none_when_base_has_none(self):
    w = workloads.ScalarScaled(alpha=2.0, base=workloads.SuffixSum(n=4))
    self.assertIsNone(w.toeplitz_coefs)

  def test_streaming_matrix_matches_dense(self):
    w = workloads.ScalarScaled(alpha=0.5, base=workloads.PrefixSum(n=6))
    sm = w.streaming_matrix
    self.assertIsNotNone(sm)
    np.testing.assert_allclose(sm.materialize(6), w.dense, atol=1e-5)

  def test_gram_equals_alpha_squared_base_gram(self):
    base = workloads.PrefixSum(n=4)
    w = workloads.ScalarScaled(alpha=3.0, base=base)
    np.testing.assert_allclose(w.gram, 9.0 * base.gram, atol=1e-5)

  def test_gram(self):
    w = workloads.ScalarScaled(alpha=3.0, base=workloads.PrefixSum(n=4))
    _assert_gram_equals_dense_gram(w)


class DiagonallyScaledTest(absltest.TestCase):

  def test_dense(self):
    scale = jnp.array([1.0, 2.0, 3.0])
    w = workloads.DiagonallyScaled(scale=scale, base=workloads.Identity(n=3))
    np.testing.assert_allclose(w.dense, jnp.diag(scale), atol=1e-6)

  def test_n_from_base(self):
    w = workloads.DiagonallyScaled(
        scale=jnp.ones(5), base=workloads.PrefixSum(n=5)
    )
    self.assertEqual(w.n, 5)

  def test_scale_length_mismatch_raises(self):
    with self.assertRaises(ValueError):
      workloads.DiagonallyScaled(
          scale=jnp.ones(3), base=workloads.PrefixSum(n=4)
      )

  def test_toeplitz_coefs_always_none(self):
    w = workloads.DiagonallyScaled(
        scale=2.0 * jnp.ones(4), base=workloads.PrefixSum(n=4)
    )
    self.assertIsNone(w.toeplitz_coefs)

  def test_gram(self):
    w = workloads.DiagonallyScaled(
        scale=jnp.array([1.0, 2.0, 3.0]),
        base=workloads.PrefixSum(n=3),
    )
    _assert_gram_equals_dense_gram(w)


class StackedTest(parameterized.TestCase):

  def test_dense(self):
    w1 = workloads.PrefixSum(n=3)
    w2 = workloads.Identity(n=3)
    w = workloads.Stacked(w1, w2)
    expected = jnp.concatenate([w1.dense, w2.dense], axis=0)
    np.testing.assert_allclose(w.dense, expected, atol=1e-6)

  def test_shape(self):
    w = workloads.Stacked(workloads.PrefixSum(n=4), workloads.SuffixSum(n=4))
    self.assertEqual(w.dense.shape, (8, 4))

  def test_n_from_children(self):
    w = workloads.Stacked(workloads.PrefixSum(n=5), workloads.Identity(n=5))
    self.assertEqual(w.n, 5)

  def test_mismatched_n_raises(self):
    with self.assertRaises(ValueError):
      workloads.Stacked(workloads.PrefixSum(n=3), workloads.Identity(n=5))

  def test_fewer_than_two_raises(self):
    with self.assertRaises(ValueError):
      workloads.Stacked(workloads.PrefixSum(n=3))

  def test_gram_is_sum_of_grams(self):
    w1 = workloads.PrefixSum(n=4)
    w2 = workloads.Identity(n=4)
    w = workloads.Stacked(w1, w2)
    np.testing.assert_allclose(w.gram, w1.gram + w2.gram, atol=1e-5)

  def test_gram(self):
    w = workloads.Stacked(workloads.PrefixSum(n=4), workloads.Identity(n=4))
    _assert_gram_equals_dense_gram(w)

  def test_regularized_prefix_sum(self):
    """The key use case: [A; lambda * I]."""
    n = 5
    lam = 0.1
    w = workloads.Stacked(
        workloads.PrefixSum(n=n),
        workloads.ScalarScaled(alpha=lam, base=workloads.Identity(n=n)),
    )
    self.assertEqual(w.dense.shape, (2 * n, n))
    # Gram = A^T A + lambda^2 * I.
    expected_gram = jnp.tri(n).T @ jnp.tri(n) + lam**2 * jnp.eye(n)
    np.testing.assert_allclose(w.gram, expected_gram, atol=1e-5)

  def test_three_workloads(self):
    n = 4
    w = workloads.Stacked(
        workloads.PrefixSum(n=n),
        workloads.SuffixSum(n=n),
        workloads.Identity(n=n),
    )
    self.assertEqual(w.dense.shape, (3 * n, n))


if __name__ == '__main__':
  absltest.main()
