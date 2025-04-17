# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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
import chex
import jax
from jax import numpy as jnp
from jax_privacy.stream_privatization import additive_privatizers
import numpy as np


_MATRIX_SIZE = 10
_VECTOR_SHAPE = (_MATRIX_SIZE,)
_MATRIX_SHAPE = (_MATRIX_SIZE, _MATRIX_SIZE)


class DenseMatMulTest(chex.TestCase, parameterized.TestCase):

  def test_matrix_multiply_raises_non_matrix_arg(self):
    with self.assertRaisesRegex(ValueError, 'Expected matrix'):
      additive_privatizers.concrete_correlated_gaussian_privatizer(
          noise_correlation_matrix=jnp.ones(shape=[10]),
          noise_key=jax.random.PRNGKey(10),
          stddev=0.0,
      )

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_matrix_multiply_identity_matrix(self, example_grad):
    noise_key = jax.random.PRNGKey(99)
    stddev = 2.0
    matmul_privatizer = (
        additive_privatizers.concrete_correlated_gaussian_privatizer(
            noise_correlation_matrix=jnp.eye(_MATRIX_SIZE),
            noise_key=noise_key,
            stddev=stddev,
        )
    )
    state = matmul_privatizer.init(example_grad)
    grads = jax.tree.map(jnp.zeros_like, example_grad)

    # With the identity matrix, we should directly add the apprporiate noise to
    # the grads on each step. So we can compute this noise by directly calling
    # the underlying isotropic gaussian noise fn (even though this test is more
    # brittle than, strictly speaking, necessary).
    def noise_at_idx(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=noise_key,
          index=idx,
          stddev=stddev,
          example_grad=example_grad,
      )

    for idx in range(_MATRIX_SIZE):
      result, state = matmul_privatizer.privatize(
          sum_of_clipped_grads=grads, noise_state=state, prng_key=noise_key,
      )
      # The identity matrix should just pick out the slice corresponding to the
      # state index,
      chex.assert_trees_all_close(result, noise_at_idx(idx))

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_matrix_multiply_dense_matrix(self, example_grad):
    noise_key = jax.random.PRNGKey(101)
    stddev = 10.0
    matmul_privatizer = (
        additive_privatizers.concrete_correlated_gaussian_privatizer(
            noise_correlation_matrix=jnp.ones(
                shape=[_MATRIX_SIZE, _MATRIX_SIZE]
            ),
            noise_key=noise_key,
            stddev=stddev,
        )
    )
    grads = jax.tree.map(jnp.zeros_like, example_grad)
    state = matmul_privatizer.init(grads)

    def noise_at_idx(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=noise_key,
          index=idx,
          stddev=stddev,
          example_grad=example_grad,
      )

    # The matrix with which we multiple here is constant 1s everywhere, so the
    # results of all

    def compute_reduced_sum_of_elements(idx):
      del idx  # Unused
      noise_accum = jax.tree.map(jnp.zeros_like, noise_at_idx(0))
      for i in range(_MATRIX_SIZE):
        noise_accum = jax.tree.map(
            lambda x, y: x + y, noise_accum, noise_at_idx(i)
        )
      return noise_accum

    for idx in range(_MATRIX_SIZE):
      result, state = matmul_privatizer.privatize(
          sum_of_clipped_grads=grads, noise_state=state, prng_key=noise_key,
      )
      chex.assert_trees_all_close(
          result,
          compute_reduced_sum_of_elements(idx),
          rtol=1e-5,
      )

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_matrix_multiply_zero_matrix(self, example_grad):
    noise_key = jax.random.PRNGKey(102)
    matmul_privatizer = (
        additive_privatizers.concrete_correlated_gaussian_privatizer(
            noise_correlation_matrix=jnp.zeros(
                shape=[_MATRIX_SIZE, _MATRIX_SIZE]
            ),
            noise_key=noise_key,
            stddev=10.0,
        )
    )
    grads = jax.tree.map(jnp.ones_like, example_grad)
    state = matmul_privatizer.init(grads)

    for _ in range(_MATRIX_SIZE):
      result, state = matmul_privatizer.privatize(
          sum_of_clipped_grads=grads, noise_state=state, prng_key=noise_key,
      )
      # Since the matrix is all zeros, the multiplication should result in
      # all-zero tensors. Since we add before returning, this means that we
      # should always get back grads unchanged.
      chex.assert_trees_all_close(
          result,
          grads,
      )


class GaussianNoiseTest(chex.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_functional_noise_generation_on_index(self, example_tensors):
    rng = jax.random.PRNGKey(2)
    stddev = 1.0

    def noise_fn(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=rng, index=idx, stddev=stddev, example_grad=example_tensors
      )

    indices_to_test = range(1, 1000, 100)

    for index1 in indices_to_test:
      for index2 in indices_to_test:
        if index1 == index2:
          # Reinvoking the same function with the same arguments should match.
          chex.assert_trees_all_close(noise_fn(index1), noise_fn(index2))
        else:
          # However, calling with a different index should give us different
          # tensors.
          self.assertFalse(np.all(np.equal(noise_fn(index1), noise_fn(index2))))

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_different_rngs_yield_different_sequences(self, example_tensors):
    rng1 = jax.random.PRNGKey(10)
    rng2 = jax.random.PRNGKey(21)
    stddev = 1.0

    def noise_fn1(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=rng1, index=idx, stddev=stddev, example_grad=example_tensors
      )

    def noise_fn2(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=rng2, index=idx, stddev=stddev, example_grad=example_tensors
      )

    indices_to_test = range(2, 1002, 100)

    for index in indices_to_test:
      self.assertFalse(np.all(np.equal(noise_fn1(index), noise_fn2(index))))

  @parameterized.named_parameters(
      ('scalar', jax.ShapeDtypeStruct(shape=[], dtype=jnp.float32)),
      ('vector', jax.ShapeDtypeStruct(shape=_VECTOR_SHAPE, dtype=jnp.float32)),
      (
          'structure_of_matrices',
          (jax.ShapeDtypeStruct(shape=_MATRIX_SHAPE, dtype=jnp.float32),) * 2,
      ),
  )
  def test_same_rngs_yield_same_sequences(self, example_tensors):
    rng = jax.random.PRNGKey(57)
    stddev = 1.0

    def noise_fn1(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=rng, index=idx, stddev=stddev, example_grad=example_tensors
      )

    def noise_fn2(idx):
      return additive_privatizers.isotropic_gaussian_noise_fn(
          rng=rng, index=idx, stddev=stddev, example_grad=example_tensors
      )

    indices_to_test = range(3, _MATRIX_SIZE + 3, 100)

    for index in indices_to_test:
      chex.assert_trees_all_close(noise_fn1(index), noise_fn2(index))

  def test_different_noise_generated_in_elements_of_structure(self):
    example_tensors = tuple(jnp.zeros(shape=[10, 10]) for _ in range(5))
    rng = jax.random.PRNGKey(57)
    stddev = 1.0
    index = 40

    random_tensors = additive_privatizers.isotropic_gaussian_noise_fn(
        rng=rng, index=index, stddev=stddev, example_grad=example_tensors
    )

    for idx1, tensor1 in enumerate(random_tensors):
      for idx2, tensor2 in enumerate(random_tensors):
        if idx1 == idx2:
          chex.assert_trees_all_close(tensor1, tensor2)
        else:
          # No simply assert not all close in chex. Drop to this spelling in
          # numpy.
          self.assertFalse(np.all(np.equal(tensor1, tensor2)))


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
