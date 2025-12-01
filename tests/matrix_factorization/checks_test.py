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

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax.numpy as jnp
from jax_privacy.matrix_factorization import checks


class ChecksTest(parameterized.TestCase):

  @hypothesis.settings(deadline=None, max_examples=10)
  @hypothesis.given(n=st.integers(2, 10))
  def test_X(self, n):
    with self.assertRaisesRegex(ValueError, 'X.*symmetric'):
      checks.check(X=jnp.tri(n))

    checks.check(X=jnp.eye(n))
    checks.check(X=jnp.ones(shape=(n, n)))

  @parameterized.named_parameters(
      ('A', lambda A: checks.check(A=A), 'A'),
      ('B', lambda B: checks.check(B=B), 'B'),
      ('C', lambda C: checks.check(C=C), 'C'),
  )
  def test_check_square_lower_triangular(self, check_fn, matrix_name):
    with self.assertRaisesRegex(ValueError, f'{matrix_name}.*finite'):
      check_fn(jnp.array([[jnp.nan]]))

    for n in [2, 3, 5]:
      with self.assertRaisesRegex(
          ValueError, f'{matrix_name}.*lower-triangular'
      ):
        check_fn(jnp.ones(shape=(n, n)))

      # No errors
      check_fn(jnp.eye(n))
      check_fn(jnp.tri(n))

    # Special case n=1 is fine.
    check_fn(jnp.array([[1]]))

  def test_B_C_inner_shape_mismatch(self):
    checks.check(B=jnp.ones(shape=(5, 7)), C=jnp.ones(shape=(7, 5)))
    with self.assertRaisesRegex(ValueError, 'B and C shapes do not match'):
      checks.check(B=jnp.ones(shape=(5, 7)), C=jnp.ones(shape=(8, 5)))

  @hypothesis.settings(deadline=None, max_examples=10)
  @hypothesis.given(n=st.integers(1, 5))
  def test_check_all_good(self, n):
    checks.check(A=jnp.tri(n), B=jnp.tri(n), C=jnp.eye(n), X=jnp.eye(n))

  @hypothesis.settings(deadline=None, max_examples=10)
  @hypothesis.given(i=st.integers(0, 3), n=st.integers(1, 5))
  def test_check_n_mismatch(self, i, n):
    n_list = [n, n, n, n]
    n_list[i] = n + 1  # One matrix has a different n value.
    with self.assertRaisesRegex(ValueError, 'Expected matrix shapes'):
      checks.check(
          A=jnp.tri(n_list[0]),
          # Make sure B and C mismatch on `n` but not the inner dimension.
          B=jnp.ones(shape=(n_list[1], 10)),
          C=jnp.ones(shape=(10, n_list[2])),
          X=jnp.eye(n_list[3]),
      )


if __name__ == '__main__':
  absltest.main()
