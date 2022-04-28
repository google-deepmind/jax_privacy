# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

"""Unit test for grad_clipping."""

import functools

from absl.testing import absltest
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.src.training import grad_clipping
import optax


NUM_SAMPLES = 4
NUM_CLASSES = 7
INPUT_SHAPE = (12, 12, 3)
MAX_NORM = 1e-4


def model_fn(inputs, num_classes):
  """Mini ConvNet."""
  out = inputs
  out = hk.Conv2D(4, 3)(inputs)
  out = jax.nn.relu(out)
  out = hk.Conv2D(4, 3)(out)
  out = jax.nn.relu(out)
  out = jnp.mean(out, axis=[1, 2])
  out = hk.Linear(num_classes)(out)
  return out


def grad_clipped_per_sample_naive(forward_fn, clipping_norm):
  """Naive implementation for computing gradients clipped per-example."""
  grad_fn = jax.grad(forward_fn, has_aux=True)

  def accumulate(tree_acc, tree_new, coeff):
    return jax.tree_map(
        lambda leaf_acc, leaf_new: leaf_acc + leaf_new * coeff,
        tree_acc,
        tree_new,
    )

  def clipped_grad_fn(params, inputs, network_state, rng_key):
    value, aux = forward_fn(params, inputs, network_state, rng_key)
    images, labels = inputs
    batch_size = len(images)
    grads = jax.tree_map(jnp.zeros_like, params)

    # compute one clipped gradient at a time
    for i in range(batch_size):
      # expand image: function expects a batch dimension
      input_i = (jnp.expand_dims(images[i], 0), labels[i])
      grad_i, unused_aux = grad_fn(params, input_i, network_state, rng_key)

      norm_grad_i = jnp.sqrt(
          sum(jnp.sum(x ** 2) for x in jax.tree_leaves(grad_i)))

      # multiplicative factor equivalent to clipping norm
      coeff = jnp.minimum(1, clipping_norm / norm_grad_i) / batch_size

      # normalize by batch_size and accumulate
      grads = accumulate(grads, grad_i, coeff)

    return (jnp.mean(value), aux), grads
  return clipped_grad_fn


class TestClippedGradients(chex.TestCase):
  """Check numerically that gradients are correctly clipped."""

  def setUp(self):
    super().setUp()

    rng_seq = hk.PRNGSequence(0)

    images = jax.random.normal(
        next(rng_seq), shape=(NUM_SAMPLES,) + INPUT_SHAPE)
    labels = jax.random.randint(
        next(rng_seq), shape=[NUM_SAMPLES], minval=0, maxval=NUM_CLASSES)
    labels_one_hot = hk.one_hot(labels, NUM_CLASSES)

    self.net = hk.transform_with_state(
        functools.partial(model_fn, num_classes=NUM_CLASSES))
    self.params, self.network_state = self.net.init(next(rng_seq), images)

    self.inputs = (images, labels_one_hot)
    self.rng_key = next(rng_seq)

    self.tol = {'rtol': 1e-6, 'atol': 1e-6}

  def forward(self, params, inputs, state, rng_key):
    images, labels = inputs
    logits, state = self.net.apply(params, state, rng_key, images)
    loss_vector = optax.softmax_cross_entropy(logits, labels)
    aux = (logits, state, jnp.ones([2]), loss_vector)
    return jnp.mean(loss_vector), aux

  @chex.variants(with_jit=True, without_jit=True)
  def test_clipped_gradients(self):

    clipping_fn = grad_clipping.global_clipping(clipping_norm=MAX_NORM)

    grad_fn_1 = grad_clipping.value_and_clipped_grad_vectorized(
        self.forward,
        clipping_fn,
    )

    grad_fn_2 = grad_clipping.value_and_clipped_grad_loop(
        self.forward,
        clipping_fn,
    )

    grad_fn_3 = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=MAX_NORM,
    )

    grad_fn_args = (self.params, self.inputs, self.network_state, self.rng_key)
    (loss_1, aux_1), (grad_1, _) = self.variant(grad_fn_1)(*grad_fn_args)
    (loss_2, aux_2), (grad_2, _) = self.variant(grad_fn_2)(*grad_fn_args)
    (loss_3, aux_3), grad_3 = self.variant(grad_fn_3)(*grad_fn_args)

    chex.assert_trees_all_close(loss_1, loss_2, loss_3, **self.tol)
    chex.assert_trees_all_close(aux_1, aux_2, aux_3, **self.tol)
    chex.assert_trees_all_close(grad_1, grad_2, grad_3, **self.tol)

  @chex.variants(with_jit=True, without_jit=True)
  def test_standard_gradients(self):
    clipping_fn = lambda grads: (grads, {})

    grad_fn_1 = grad_clipping.value_and_clipped_grad_vectorized(
        self.forward,
        clipping_fn=clipping_fn,
    )

    grad_fn_2 = grad_clipping.value_and_clipped_grad_loop(
        self.forward,
        clipping_fn=clipping_fn,
    )

    grad_fn_3 = jax.value_and_grad(self.forward, has_aux=True)

    grad_fn_args = (self.params, self.inputs, self.network_state, self.rng_key)
    (loss_1, aux_1), (grad_1, _) = self.variant(grad_fn_1)(*grad_fn_args)
    (loss_2, aux_2), (grad_2, _) = self.variant(grad_fn_2)(*grad_fn_args)
    (loss_3, aux_3), grad_3 = self.variant(grad_fn_3)(*grad_fn_args)

    chex.assert_trees_all_close(loss_1, loss_2, loss_3, **self.tol)
    chex.assert_trees_all_close(aux_1, aux_2, aux_3, **self.tol)
    chex.assert_trees_all_close(grad_1, grad_2, grad_3, **self.tol)


if __name__ == '__main__':
  absltest.main()
