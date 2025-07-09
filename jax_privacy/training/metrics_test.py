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
import jax.numpy as jnp
from jax_privacy.training import metrics


class AccuracyTest(absltest.TestCase):

  def test_distinct(self):

    logits = jnp.array([
        [0.1, 0.5, 0.2],
        [-0.3, -0.2, 0.0],
    ])
    labels = jnp.array([1, 2])
    labels_one_hot = jax.nn.one_hot(labels, 3)
    acc1, acc2, acc3 = metrics.topk_accuracy(
        logits, labels_one_hot, topk=[1, 2, 3])

    chex.assert_equal(acc1, 1.0)  # all correct
    chex.assert_equal(acc2, 1.0)  # all correct
    chex.assert_equal(acc3, 1.0)  # all correct

  def test_with_ties(self):

    logits = jnp.array([
        [0.1, 0.5, 0.5],
        [-0.2, -0.2, -0.2],
    ])
    labels = jnp.array([1, 2])
    labels_one_hot = jax.nn.one_hot(labels, 3)

    acc1, acc2, acc3 = metrics.topk_accuracy(
        logits, labels_one_hot, topk=[1, 2, 3])

    chex.assert_equal(acc1, 0.0)  # all incorrect
    chex.assert_equal(acc2, 0.5)  # first sample correct, second one incorrect
    chex.assert_equal(acc3, 1.0)  # all correct

  def test_with_nan(self):

    logits = jnp.array([
        [0.1, 0.5, jnp.nan],
        [-0.2, -0.2, -0.2],
        [-0.3, -0.2, -0.5],
    ])
    labels = jnp.array([1, jnp.nan, 0])
    labels_one_hot = jax.nn.one_hot(labels, 3)

    acc1, acc2, acc3 = metrics.topk_accuracy(
        logits, labels_one_hot, topk=[1, 2, 3])

    chex.assert_equal(acc1, 0.0)  # all incorrect
    chex.assert_equal(acc2, 1 / 3)  # third sample correct
    chex.assert_equal(acc3, 1 / 3)  # third sample correct


if __name__ == '__main__':
  absltest.main()
