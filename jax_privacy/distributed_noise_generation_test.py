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
import chex
import jax
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.matrix_factorization import buffered_toeplitz
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz
import numpy as np


# pylint: disable=invalid-name


def empty_shardings_fn(out_sharding):
  empty_fn = lambda s: jax.sharding.NamedSharding(
      s.mesh, jax.sharding.PartitionSpec()
  )
  return jax.tree.map(empty_fn, out_sharding)


def banded_toeplitz_noising_matrix_fn():
  return toeplitz.inverse_as_streaming_matrix(jnp.array([1, 0.5, 0.3, 0.25]))


def buffered_toeplitz_noising_matrix_fn():
  blt = buffered_toeplitz.BufferedToeplitz.build(
      buf_decay=[0.9, 0.8, 0.7], output_scale=[0.1, 0.2, 0.3], dtype=jnp.float32
  )
  return blt.inverse_as_streaming_matrix()


class ShardedNoiseGenerationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(8)

    axis_types = (jax.sharding.AxisType.Explicit,) * 2
    mesh = jax.make_mesh((4, 2), ('x', 'y'), axis_types=axis_types)
    jax.sharding.set_mesh(mesh)

  @parameterized.named_parameters(
      ('blt', buffered_toeplitz_noising_matrix_fn),
      ('bandmf', banded_toeplitz_noising_matrix_fn),
      ('dpsgd', streaming_matrix.identity),
      ('prefix', streaming_matrix.prefix_sum),
      ('momentum', streaming_matrix.momentum_sgd_matrix),
  )
  def test_runs_as_expected(
      self,
      strategy_inverse_fn=banded_toeplitz_noising_matrix_fn,
  ):
    self.assertEqual(jax.device_count(), 8)

    pspecs = {
        'v': jax.sharding.PartitionSpec(),
        'w': jax.sharding.PartitionSpec(None, 'x'),
        'x': jax.sharding.PartitionSpec(('x', 'y')),
        'y': jax.sharding.PartitionSpec(None, None, None),
        'z': jax.sharding.PartitionSpec('y', None),
    }

    model_params = {
        'v': jnp.array(123.456),  # scalar
        'w': jnp.zeros((4, 8)),  # matrix, size divisible by 8
        'x': jnp.arange(16).astype(float),  # vector, length divisible by 8
        'y': jax.random.normal(jax.random.key(0), (2, 3, 4)),  # rank 3 tensor
        'z': jnp.ones((2, 7)),  # matrix, size not divisible by 8
    }

    model_params = jax.sharding.reshard(model_params, pspecs)

    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix=strategy_inverse_fn(),
        stddev=1.0,
        prng_key=jax.random.key(0),
        intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
    )

    @jax.jit
    def foo(model_params):
      noise_state = privatizer.init(model_params)
      noisy_grads, noise_state = privatizer.update(model_params, noise_state)
      noisy_grads2, _ = privatizer.update(model_params, noise_state)
      return noisy_grads, noisy_grads2

    model_params = jax.sharding.reshard(model_params, pspecs)
    noisy_grads, noisy_grads2 = foo(model_params)

    def assert_shape_dtype_sharding_equal(x, y):
      self.assertEqual(x.shape, y.shape)
      self.assertEqual(x.dtype, y.dtype)
      self.assertTrue(x.sharding.is_equivalent_to(y.sharding, y.ndim))

    jax.tree.map(assert_shape_dtype_sharding_equal, model_params, noisy_grads)

    # All noise is unique, both across iterations and across array elements.
    flat_noise = np.concatenate(
        [np.array(x).flatten() for x in jax.tree.flatten(noisy_grads)[0]]
        + [np.array(x).flatten() for x in jax.tree.flatten(noisy_grads2)[0]]
    )
    self.assertLen(set(flat_noise), flat_noise.size)

  def test_internal_shardings(self):

    noising_matrix = toeplitz.inverse_as_streaming_matrix(
        jnp.ones(3, dtype=jnp.float32)
    )
    privatizer = noise_addition.matrix_factorization_privatizer(
        noising_matrix=noising_matrix,
        stddev=1.0,
        prng_key=jax.random.key(0),
        intermediate_strategy=noise_addition.SupportedStrategies.ZERO,
    )

    def foo(sum_of_clipped_grads):
      noise_state0 = privatizer.init(sum_of_clipped_grads)
      _, noise_state1 = privatizer.update(sum_of_clipped_grads, noise_state0)
      _, noise_state2 = privatizer.update(sum_of_clipped_grads, noise_state1)
      return noise_state0, noise_state1, noise_state2

    expected = jax.sharding.PartitionSpec(None, ('x', 'y'))
    params = jax.device_put(
        jnp.zeros((3, 4, 5), dtype=jnp.float32), jax.sharding.PartitionSpec()
    )
    states = foo(params)

    for full_state in states:
      _, state = full_state
      print(state)
      self.assertEqual(state.shape[1], 64)
      self.assertEqual(state.dtype, jnp.float32)
      print(state.sharding)
      self.assertEqual(state.sharding.spec, expected)


if __name__ == '__main__':
  absltest.main()
