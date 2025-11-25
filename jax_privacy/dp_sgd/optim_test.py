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
import chex
import jax
from jax_privacy.dp_sgd import optim


class TestTreeMapAddNormalNoise(chex.TestCase):
  """Test whether noisy inputs differ from inputs with and without noise."""

  def setUp(self):
    super().setUp()

    self.rng_key = jax.random.PRNGKey(0)
    self.noise_std = 0.1

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(1), 3)
    self.tree = {'a': jax.random.normal(key1, (2, 2)),
                 'b': [jax.random.normal(key2, (1, 2)),
                       jax.random.normal(key3, ())]}

  def test_with_noise(self):
    noisy_tree = optim.tree_map_add_normal_noise(
        self.tree, self.noise_std, self.rng_key)

    self.assertEqual(jax.tree_util.tree_structure(noisy_tree),
                     jax.tree_util.tree_structure(self.tree))
    with self.assertRaises(AssertionError):
      chex.assert_trees_all_close(self.tree, noisy_tree)

  def test_without_noise(self):
    tree_with_zero_noise = optim.tree_map_add_normal_noise(
        self.tree, 0.0, self.rng_key)

    self.assertEqual(jax.tree_util.tree_structure(tree_with_zero_noise),
                     jax.tree_util.tree_structure(self.tree))
    chex.assert_trees_all_close(self.tree, tree_with_zero_noise)


if __name__ == '__main__':
  absltest.main()
