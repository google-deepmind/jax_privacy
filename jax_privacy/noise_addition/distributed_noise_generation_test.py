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

import math

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
# pylint: disable=g-importing-member
from jax.experimental.shard import reshard
from jax_privacy.matrix_factorization import buffered_toeplitz
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz
from jax_privacy.noise_addition import distributed_noise_generation
import numpy as np


# pylint: disable=invalid-name


def flatten_shardings_fn(out_sharding):
  flatten_fn = lambda s: jax.sharding.NamedSharding(
      s.mesh, distributed_noise_generation._flatten_pspec(s.spec)
  )
  return jax.tree.map(flatten_fn, out_sharding)


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


def get_debug_streaming_matrix(
    num_buffers: int = 3,
) -> streaming_matrix.StreamingMatrix:
  """Returns a streaming matrix useful for testing."""

  def _test_init(shape: tuple[int, ...]) -> jax.Array:
    mat_shape = (num_buffers,) + shape
    num_entries = math.prod(mat_shape)
    return jnp.arange(num_entries, dtype=jnp.float32).reshape(mat_shape)

  def _test_multiply(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.sum(x * y, axis=0), y

  return streaming_matrix.StreamingMatrix(_test_init, _test_multiply)


class ShardedNoiseGenerationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(8)

  def test_flatten_pspec(self):
    self.assertEqual(
        distributed_noise_generation._flatten_pspec(
            jax.sharding.PartitionSpec(None, ('x', 'y'), None, 'z')
        ),
        jax.sharding.PartitionSpec(('x', 'y', 'z')),
    )
    self.assertEqual(
        distributed_noise_generation._flatten_pspec(
            jax.sharding.PartitionSpec('data', None, ('replica', 'mdl'))
        ),
        jax.sharding.PartitionSpec(('data', 'replica', 'mdl')),
    )

  def test_reshape_add(self):
    axis_types = (jax.sharding.AxisType.Explicit,) * 3
    mesh = jax.make_mesh((2, 2, 2), ('x', 'y', 'z'), axis_types=axis_types)

    with jax.sharding.use_mesh(mesh):
      x = reshard(
          jnp.zeros((2, 4, 8)), jax.sharding.PartitionSpec(None, 'y', 'x')
      )
      y = reshard(
          jnp.arange(2 * 4 * 8), jax.sharding.PartitionSpec(('x', 'y', 'z'))
      )
      z = distributed_noise_generation._reshape_add(x, y)

    self.assertEqual(z.shape, x.shape)
    self.assertEqual(z.sharding, x.sharding)

    actual = np.sort(np.array(z).flatten())
    chex.assert_trees_all_equal(actual, y)

  @parameterized.named_parameters(
      ('dpsgd', streaming_matrix.identity),
      ('prefix', streaming_matrix.prefix_sum),
      ('momentum', streaming_matrix.momentum_sgd_matrix),
  )
  def test_runs_as_expected(
      self,
      strategy_inverse_fn=banded_toeplitz_noising_matrix_fn,
  ):
    self.assertEqual(jax.device_count(), 8)

    axis_types = (jax.sharding.AxisType.Explicit,) * 2
    mesh = jax.make_mesh((4, 2), ('x', 'y'), axis_types=axis_types)

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

    with jax.sharding.use_mesh(mesh):
      model_params = reshard(model_params, pspecs)
      noising_matrix = strategy_inverse_fn()

    privatizer = (
        distributed_noise_generation.streaming_matrix_to_sharded_privatizer(
            noising_matrix=noising_matrix,
            stddev=1.0,
            noise_key=jax.random.key(0),
        )
    )

    @jax.jit
    def foo(model_params):
      noise_state = privatizer.init(model_params)
      sum_of_clipped_grads = model_params
      noisy_grads, noise_state = privatizer.privatize(
          sum_of_clipped_grads=sum_of_clipped_grads, noise_state=noise_state
      )
      noisy_grads2, _ = privatizer.privatize(
          sum_of_clipped_grads=sum_of_clipped_grads, noise_state=noise_state
      )
      return noisy_grads, noisy_grads2

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
    axis_types = (jax.sharding.AxisType.Explicit,) * 2
    mesh = jax.make_mesh((4, 2), ('x', 'y'), axis_types=axis_types)

    privatizer = (
        distributed_noise_generation.streaming_matrix_to_sharded_privatizer(
            noising_matrix=get_debug_streaming_matrix(),
            stddev=1.0,
            noise_key=jax.random.key(0),
        )
    )

    def foo(sum_of_clipped_grads):
      noise_state0 = privatizer.init(sum_of_clipped_grads)
      _, noise_state1 = privatizer.privatize(
          sum_of_clipped_grads=sum_of_clipped_grads,
          noise_state=noise_state0,
      )
      _, noise_state2 = privatizer.privatize(
          sum_of_clipped_grads=sum_of_clipped_grads,
          noise_state=noise_state1,
      )
      return noise_state0, noise_state1, noise_state2

    expected = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, ('x', 'y'))
    )
    with jax.sharding.use_mesh(mesh):
      params = jax.device_put(
          jnp.zeros((3, 4, 5)), jax.sharding.PartitionSpec()
      )
      states = foo(params)

    for full_state in states:
      _, state = full_state
      self.assertEqual(state.shape[1], 64)
      self.assertEqual(state.dtype, jnp.float32)
      self.assertTrue(state.sharding.is_equivalent_to(expected, state.ndim))


if __name__ == '__main__':
  absltest.main()
