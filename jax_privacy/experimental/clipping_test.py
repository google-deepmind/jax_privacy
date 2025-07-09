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
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy.experimental import clipping
import optax


class ClipPyTreeTest(absltest.TestCase):

  def test_clip_pytree_clips_when_norm_exceeds_max(self):
    pytree = {'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}
    clip_norm = 4.0

    clipped_tree, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=False
    )
    chex.assert_trees_all_close(
        clipped_tree, {'a': jnp.array([2.4, 0.0]), 'b': jnp.array([0.0, 3.2])}
    )
    self.assertAlmostEqual(norm, 5.0)
    self.assertAlmostEqual(optax.global_norm(clipped_tree), clip_norm)

    clipped_tree_rescaled, norm_rescaled = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=True
    )
    chex.assert_trees_all_close(
        clipped_tree_rescaled,
        {'a': jnp.array([0.6, 0.0]), 'b': jnp.array([0.0, 0.8])},
    )
    self.assertAlmostEqual(norm_rescaled, 5.0)
    self.assertAlmostEqual(optax.global_norm(clipped_tree_rescaled), 1.0)

  def test_clip_pytree_does_not_clip_when_norm_below_max(self):
    pytree = {'a': jnp.array([1.0, 0.0]), 'b': jnp.array([0.0, 1.0])}
    clip_norm = 2.0

    clipped_tree, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=False
    )
    chex.assert_trees_all_close(clipped_tree, pytree)
    self.assertAlmostEqual(norm, jnp.sqrt(2.0))

    clipped_tree_rescaled, norm_rescaled = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=True
    )
    expected_rescaled = jax.tree.map(lambda x: x / clip_norm, pytree)
    chex.assert_trees_all_close(clipped_tree_rescaled, expected_rescaled)
    self.assertAlmostEqual(norm_rescaled, jnp.sqrt(2.0))
    self.assertAlmostEqual(
        optax.global_norm(clipped_tree_rescaled),
        jnp.sqrt(2.0) / clip_norm,
    )

  def test_clip_pytree_clip_norm_zero(self):
    """Tests clip_pytree when clip_norm is 0."""
    pytree = {'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}
    zero_tree = jax.tree.map(jnp.zeros_like, pytree)
    clip_norm = 0.0

    clipped, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=False
    )
    self.assertAlmostEqual(norm, 5.0)
    chex.assert_trees_all_close(clipped, zero_tree)
    self.assertAlmostEqual(optax.global_norm(clipped), 0.0)

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

  def test_clip_pytree_clip_norm_inf(self):
    """Tests clip_pytree when clip_norm is infinity."""
    pytree = {'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}
    zero_tree = jax.tree.map(jnp.zeros_like, pytree)
    clip_norm = jnp.inf

    clipped, norm = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=False
    )
    self.assertAlmostEqual(norm, 5.0)
    chex.assert_trees_all_close(clipped, pytree)
    self.assertAlmostEqual(optax.global_norm(clipped), 5.0)

    clipped_rescaled, norm_rescaled = clipping.clip_pytree(
        pytree, clip_norm, rescale_to_unit_norm=True
    )
    self.assertAlmostEqual(norm_rescaled, 5.0)
    chex.assert_trees_all_close(clipped_rescaled, zero_tree)
    self.assertAlmostEqual(optax.global_norm(clipped_rescaled), 0.0)

  def test_clip_pytree_zero_norm_pytree(self):
    """Tests clip_pytree when the input pytree has zero norm."""
    zero_tree = {'a': jnp.array([0.0, 0.0]), 'b': jnp.array([0.0, 0.0])}
    clip_norm = 5.0

    clipped, norm = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=False
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

  def test_clip_pytree_zero_norm_and_zero_clip(self):
    """Tests clip_pytree when both input norm and clip_norm are zero."""
    zero_tree = {'a': jnp.array([0.0, 0.0]), 'b': jnp.array([0.0, 0.0])}
    clip_norm = 0.0

    clipped, norm = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=False
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

  def test_clip_pytree_zero_norm_and_inf_clip(self):
    """Tests clip_pytree when input norm is zero and clip_norm is inf."""
    zero_tree = {'a': jnp.array([0.0, 0.0]), 'b': jnp.array([0.0, 0.0])}
    clip_norm = jnp.inf

    clipped, norm = clipping.clip_pytree(
        zero_tree, clip_norm, rescale_to_unit_norm=False
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
    sum_clip_mean = clipping.clip_sum(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

    is_padding_example_neighbor = jnp.zeros(data.shape[0]).at[3].set(1)
    relation = dp_accounting.NeighboringRelation.REPLACE_SPECIAL
    new = sum_clip_mean(data, is_padding_example=is_padding_example_neighbor)
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
    sum_clip_mean = clipping.clip_sum(**kwargs)

    neighbor = data.at[3].set([-10, 10, 20, 20])
    diff = jnp.linalg.norm(sum_clip_mean(data) - sum_clip_mean(neighbor))
    relation = dp_accounting.NeighboringRelation.REPLACE_ONE
    self.assertLessEqual(diff, sum_clip_mean.sensitivity(relation) + 1e-6)

  @parameterized.product(
      arg_dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      output_dtype=[jnp.float32, None],
      microbatch_size=[3, None],
  )
  def test_correct_dtype(self, arg_dtype, output_dtype, microbatch_size):
    data = jax.random.normal(jax.random.key(0), (15, 4), arg_dtype)
    sum_clip_mean = clipping.clip_sum(
        **CLIP_SUM_INPUTS[0],
        dtype=output_dtype,
        microbatch_size=microbatch_size
    )
    value = sum_clip_mean(data)
    expected_dtype = output_dtype or arg_dtype
    self.assertEqual(value.dtype, expected_dtype)


if __name__ == '__main__':
  absltest.main()
