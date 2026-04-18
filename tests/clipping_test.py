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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import clipping
import optax


def cartesian_product(**kwargs):
  return [dict(zip(kwargs, v)) for v in itertools.product(*kwargs.values())]


PYTREE_STRUCTS = [
    jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
    {
        'a': jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.float16),
        'b': jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        'c': jax.ShapeDtypeStruct(shape=(), dtype=jnp.bfloat16),
    },
]


class ClipPyTreeTest(parameterized.TestCase):

  @parameterized.parameters(
      *cartesian_product(
          pytree=PYTREE_STRUCTS,
          clip_norm=[0.0, 1.0, 2.0, jnp.inf],
          rescale_to_unit_norm=[True, False],
          nan_safe=[True, False],
          return_zero=[True, False],
      )
  )
  def test_clip_pytree_output_dtype_matches_input(self, pytree, **kwargs):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clipped, _ = clipping.clip_pytree(pytree, **kwargs)

    jitted = jax.jit(clipping.clip_pytree, static_argnums=(2, 3))
    clipped2, _ = jitted(pytree, **kwargs)
    chex.assert_trees_all_equal_shapes_and_dtypes(pytree, clipped, clipped2)

  @parameterized.parameters(
      *cartesian_product(
          pytree=PYTREE_STRUCTS,
          clip_norm=[0.0, 1.0, 2.0, jnp.inf],
          rescale_to_unit_norm=[True, False],
          nan_safe=[True, False],
          return_zero=[True, False],
      )
  )
  def test_clip_pytree_has_bounded_norm(self, pytree, **kwargs):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clipped, _ = clipping.clip_pytree(pytree, **kwargs)

    if kwargs['return_zero']:
      chex.assert_trees_all_close(clipped, optax.tree.zeros_like(pytree))
    if kwargs['rescale_to_unit_norm']:
      self.assertLessEqual(optax.global_norm(clipped), 1.0)
    else:
      self.assertLessEqual(optax.global_norm(clipped), kwargs['clip_norm'])

  @parameterized.parameters(*cartesian_product(pytree=PYTREE_STRUCTS))
  def test_clip_pytree_with_large_clip_norm(self, pytree):
    pytree = optax.tree.random_like(jax.random.key(0), pytree)
    clip_norm = optax.global_norm(pytree) * 1.5
    clipped_tree, _ = clipping.clip_pytree(pytree, clip_norm)
    chex.assert_trees_all_close(clipped_tree, pytree, atol=1e-6)

  def test_clip_pytree_clip_norm_zero_rescale(self):
    """Tests clip_pytree when clip_norm is 0."""
    pytree = {'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}
    clip_norm = 0.0

    clipped, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=True
    )
    self.assertAlmostEqual(norm, 5.0)
    self.assertAlmostEqual(optax.global_norm(clipped), 1.0)
    # Normalization is the limiting behavior as clip_norm -> 0.
    expected = {'a': jnp.array([0.6, 0.0]), 'b': jnp.array([0.0, 0.8])}
    chex.assert_trees_all_close(clipped, expected)

  @parameterized.parameters(
      *cartesian_product(pytree=PYTREE_STRUCTS, clip_norm=[0.0, 1.0, jnp.inf])
  )
  def test_clip_pytree_zero_norm_pytree(self, pytree, clip_norm):
    """Tests clip_pytree when the input pytree has zero norm."""
    zero_tree = optax.tree.zeros_like(pytree)

    clipped, norm = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=False, nan_safe=True
    )
    self.assertAlmostEqual(norm, 0.0)
    chex.assert_trees_all_close(clipped, zero_tree)
    self.assertAlmostEqual(optax.global_norm(clipped), 0.0)

    clipped_rescaled, norm_rescaled = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=True
    )
    self.assertAlmostEqual(norm_rescaled, 0.0)
    chex.assert_trees_all_close(clipped_rescaled, zero_tree)
    self.assertAlmostEqual(optax.global_norm(clipped_rescaled), 0.0)

  @parameterized.parameters(
      *cartesian_product(
          clip_norm=[0.0, 1.0, 2.0, jnp.inf],
          return_zero=[False, True],
          rescale_to_unit_norm=[False, True],
      )
  )
  def test_nan_safety(self, **kwargs):
    pytree = jnp.array([3.0, 0.0, 1.0, jnp.nan, 2.0])
    clipped, _ = clipping.clip_pytree(pytree, nan_safe=True, **kwargs)
    chex.assert_tree_all_finite(clipped)

    pytree = jnp.array([3.0, 0.0, jnp.inf, -jnp.inf, 2.0])
    clipped, _ = clipping.clip_pytree(pytree, nan_safe=True, **kwargs)
    chex.assert_tree_all_finite(clipped)

  def test_clip_pytree_with_nan(self):
    """Tests clip_pytree when the input pytree has nan."""
    array = jnp.array([3.0, 0.0, 1.0, jnp.nan, 2.0])
    array_no_nan = jnp.array([3.0, 0.0, 1.0, 0.0, 2.0])
    clip_norm = 1.0
    clipped = clipping.clip_pytree(array, clip_norm)[0]
    self.assertLessEqual(jnp.linalg.norm(clipped), clip_norm)

    expected = clipping.clip_pytree(array_no_nan, clip_norm)[0]
    chex.assert_trees_all_close(clipped, expected)

  def test_clip_pytree_with_inf(self):
    """Tests clip_pytree when the input pytree has nan."""
    pytree = jnp.array([3.0, 0.0, jnp.inf, -jnp.inf, 2.0])
    pytree_no_inf = jnp.array([3.0, 0.0, 0.0, 0.0, 2.0])
    clip_norm = 1.0
    clipped = clipping.clip_pytree(pytree, clip_norm)[0]
    expected = clipping.clip_pytree(pytree_no_inf, clip_norm)[0]
    chex.assert_trees_all_close(clipped, expected)

    self.assertLessEqual(jnp.linalg.norm(clipped), clip_norm)


def single_example_fun(data):
  assert data.ndim == 1
  return data + 2


def batched_fun(data):
  assert data.ndim == 2
  return data.mean(axis=0)


CLIP_SUM_INPUTS = [
    {'fun': single_example_fun, 'batch_argnums': 0, 'keep_batch_dim': False},
    {'fun': batched_fun, 'batch_argnums': 0},
]


class ClipTransformTest(parameterized.TestCase):

  @parameterized.parameters(CLIP_SUM_INPUTS)
  def test_clip_sum_sensitivity(self, **kwargs):
    data = jax.random.normal(jax.random.key(0), (15, 4))
    sum_clip_mean = clipping.clipped_fun(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

    is_padding_example_nbr = jnp.zeros(data.shape[0], jnp.bool_).at[3].set(1)
    relation = dp_accounting.NeighboringRelation.REPLACE_SPECIAL
    new = sum_clip_mean(data, is_padding_example=is_padding_example_nbr)
    diff = jnp.linalg.norm(sum_clip_mean(data) - new)
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

    neighbor = data[1:]
    relation = dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

  @parameterized.parameters(CLIP_SUM_INPUTS)
  def test_nan_safe_by_default(self, **kwargs):
    data = jax.random.normal(jax.random.key(0), (15, 4))
    data = data.at[5].set(jnp.nan)
    sum_clip_mean = clipping.clipped_fun(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

  def test_return_norms_without_aux_matches_documented_signature(self):

    def fun(x):
      return x**2

    sum_clip = clipping.clipped_fun(
        fun,
        batch_argnums=0,
        keep_batch_dim=False,
        l2_clip_norm=jnp.inf,
        return_norms=True,
    )

    value, norms = sum_clip(jnp.array([1.0, 2.0, 3.0]))
    chex.assert_trees_all_close(value, 14.0)
    chex.assert_trees_all_close(norms, jnp.array([1.0, 4.0, 9.0]))

  @parameterized.product(
      arg_dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      output_dtype=[jnp.float32, None],
      microbatch_size=[3, None],
  )
  def test_correct_dtype(self, arg_dtype, output_dtype, microbatch_size):
    data = jax.random.normal(jax.random.key(0), (15, 4), arg_dtype)
    sum_clip_mean = clipping.clipped_fun(
        **CLIP_SUM_INPUTS[0],
        dtype=output_dtype,
        microbatch_size=microbatch_size
    )
    value = sum_clip_mean(data)
    expected_dtype = output_dtype or arg_dtype
    self.assertEqual(value.dtype, expected_dtype)

  def test_aux_shape_keep_batch_dim(self):
    def fun(batched_data):
      # With keep_batch_dim=True, data shape inside vmap is (1, D)
      scalar = jnp.sum(batched_data)
      array_1d = batched_data[:, 0, 0, 0]
      array_2d = batched_data[:, :, 0, 0]
      array_3d = batched_data[:, :, :, 1]
      return scalar, (scalar, array_1d, array_2d, array_3d, batched_data)

    data = jnp.zeros((15, 4, 9, 10))

    sum_clip_mean = clipping.clipped_fun(
        fun, has_aux=True, keep_batch_dim=True, batch_argnums=0
    )
    _, (aux_scalar, aux_1d, aux_2d, aux_3d, aux_4d) = sum_clip_mean(data)

    self.assertEqual(aux_scalar.shape, (15,))
    # In assertions below dimension of size 1 is dropped.
    self.assertEqual(aux_1d.shape, (15,))
    self.assertEqual(aux_2d.shape, (15, 4))
    self.assertEqual(aux_3d.shape, (15, 4, 9))
    self.assertEqual(aux_4d.shape, (15, 4, 9, 10))


if __name__ == '__main__':
  absltest.main()
