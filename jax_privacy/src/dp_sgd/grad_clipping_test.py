# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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
from jax_privacy.src.dp_sgd import grad_clipping
from jax_privacy.src.dp_sgd import typing
import optax

NUM_SAMPLES = 4
NUM_CLASSES = 7
INPUT_SHAPE = (12, 12, 3)
MAX_NORM = 1e-4


def model_fn(inputs, num_classes):
  """Mini ConvNet."""
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
    return jax.tree_util.tree_map(
        lambda leaf_acc, leaf_new: leaf_acc + leaf_new * coeff,
        tree_acc,
        tree_new,
    )

  def clipped_grad_fn(params, network_state, rng_per_example, inputs):
    loss, (network_state, metrics) = forward_fn(
        params, network_state, rng_per_example, inputs)
    images, labels = inputs
    batch_size = len(images)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad_norms = []

    # compute one clipped gradient at a time
    for i in range(batch_size):
      # expand image: function expects a batch dimension
      input_i = (jnp.expand_dims(images[i], 0), labels[i])
      grad_i, unused_aux = grad_fn(
          params, network_state, rng_per_example, input_i)

      norm_grad_i = jnp.sqrt(
          sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grad_i)))

      # multiplicative factor equivalent to clipping norm
      coeff = jnp.minimum(1, clipping_norm / norm_grad_i) / batch_size

      # normalize by batch_size and accumulate
      grads = accumulate(grads, grad_i, coeff)
      grad_norms.append(optax.global_norm(grad_i))

    metrics = metrics.replace(
        per_example={
            'grad_norm': jnp.array(grad_norms),
            **metrics.per_example,
        }
    )
    return (loss, (network_state, metrics)), grads

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
    self.rng_per_batch = next(rng_seq)
    self.rng_per_example = jax.random.fold_in(self.rng_per_batch, 1)

    self.tol = {'rtol': 1e-6, 'atol': 1e-6}

  def forward(self, params, state, rng_per_example, inputs):
    images, labels = inputs
    logits, state = self.net.apply(params, state, rng_per_example, images)
    loss_vector = optax.softmax_cross_entropy(logits, labels)
    metrics = typing.Metrics(
        per_example={'loss': loss_vector, 'a': jnp.ones([images.shape[0]])},
        scalars_avg={'loss': jnp.mean(loss_vector), 'b': jnp.ones([])},
    )
    return jnp.mean(loss_vector), (state, metrics)

  def forward_per_sample(self, params, state, rng_per_example, inputs):
    images, labels = inputs
    logits, state = self.net.apply(params, state, rng_per_example, images)
    loss_vector = optax.softmax_cross_entropy(logits, labels)
    metrics = typing.Metrics(
        per_example={'loss': loss_vector, 'a': jnp.ones([images.shape[0]])},
        scalars_avg={'loss': jnp.mean(loss_vector), 'b': jnp.ones([])},
    )
    return jnp.mean(loss_vector), (state, metrics)

  @chex.variants(with_jit=True, without_jit=True)
  def test_clipped_gradients(self):
    value_and_grad_fn = jax.value_and_grad(self.forward, has_aux=True)
    clipping_fn = grad_clipping.global_clipping(clipping_norm=MAX_NORM)

    grad_fn_1 = grad_clipping.value_and_clipped_grad_vectorized(
        value_and_grad_fn,
        clipping_fn,
    )

    grad_fn_2 = grad_clipping.value_and_clipped_grad_loop(
        value_and_grad_fn,
        clipping_fn,
    )

    grad_fn_3 = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=MAX_NORM,
    )

    grad_fn_args = (
        self.params, self.network_state, self.rng_per_example, self.inputs)
    (loss_1, aux_1), grad_1 = self.variant(grad_fn_1)(*grad_fn_args)
    (loss_2, aux_2), grad_2 = self.variant(grad_fn_2)(*grad_fn_args)
    (loss_3, aux_3), grad_3 = self.variant(grad_fn_3)(*grad_fn_args)

    chex.assert_trees_all_close(loss_1, loss_2, loss_3, **self.tol)
    chex.assert_trees_all_close(aux_1, aux_2, aux_3, **self.tol)
    chex.assert_trees_all_close(grad_1, grad_2, grad_3, **self.tol)

  @chex.variants(with_jit=True, without_jit=True)
  def test_gradients_vectorized_and_loop_match_using_batch_rng(self):
    value_and_grad_fn = jax.value_and_grad(self.forward, has_aux=True)
    clipping_fn = lambda grads: (grads, optax.global_norm(grads))

    grad_fn_1 = grad_clipping.value_and_clipped_grad_vectorized(
        value_and_grad_fn,
        clipping_fn=clipping_fn,
    )

    grad_fn_2 = grad_clipping.value_and_clipped_grad_loop(
        value_and_grad_fn,
        clipping_fn=clipping_fn,
    )

    grad_fn_3 = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=float('inf'),
    )

    grad_fn_args = (
        self.params, self.network_state, self.rng_per_example, self.inputs)
    (loss_1, aux_1), grad_1 = self.variant(grad_fn_1)(*grad_fn_args)
    (loss_2, aux_2), grad_2 = self.variant(grad_fn_2)(*grad_fn_args)
    (loss_3, aux_3), grad_3 = self.variant(grad_fn_3)(*grad_fn_args)

    chex.assert_trees_all_close(loss_1, loss_2, loss_3, **self.tol)
    chex.assert_trees_all_close(aux_1, aux_2, aux_3, **self.tol)
    chex.assert_trees_all_close(grad_1, grad_2, grad_3, **self.tol)

  @chex.variants(with_jit=True, without_jit=True)
  def test_gradients_vectorized_and_loop_match_using_per_sample_rng(self):
    clipping_fn = lambda grads: (grads, optax.global_norm(grads))

    grad_fn_1 = grad_clipping.value_and_clipped_grad_vectorized(
        jax.value_and_grad(self.forward_per_sample, has_aux=True),
        clipping_fn=clipping_fn,
    )

    grad_fn_2 = grad_clipping.value_and_clipped_grad_loop(
        jax.value_and_grad(self.forward_per_sample, has_aux=True),
        clipping_fn=clipping_fn,
    )

    grad_fn_args = (
        self.params, self.network_state, self.rng_per_example, self.inputs)
    (loss_1, aux_1), grad_1 = self.variant(grad_fn_1)(*grad_fn_args)
    (loss_2, aux_2), grad_2 = self.variant(grad_fn_2)(*grad_fn_args)

    chex.assert_trees_all_close(loss_1, loss_2, **self.tol)
    chex.assert_trees_all_close(aux_1, aux_2, **self.tol)
    chex.assert_trees_all_close(grad_1, grad_2, **self.tol)


@hk.transform_with_state
def simple_net(x):
  """A simple function that computes L = 3 * x + random noise."""
  key = hk.next_rng_key()
  noise = jax.random.normal(key, shape=x.shape)
  return jnp.mean(3 * x + noise), noise


class TestClippedGradientsPerBatchPerSampleRNG(chex.TestCase):
  """Check per-batch rng and per-sample rng are handled correctly."""

  def setUp(self):
    super().setUp()
    rng_seq = hk.PRNGSequence(0)
    self.inputs = jax.random.normal(next(rng_seq), shape=(NUM_SAMPLES, 10))
    self.params, self.state = simple_net.init(next(rng_seq), self.inputs)
    self.rng_per_batch = next(rng_seq)
    self.rng_per_example = jax.random.fold_in(self.rng_per_batch, 1)

    self.grad_fn_args = (self.params, self.state,
                         self.rng_per_example, self.inputs)
    self.no_clip = lambda grads: (grads, optax.global_norm(grads))

  @chex.variants(with_jit=True, without_jit=True)
  def test_per_sample_rng_produces_different_random_numbers(self):

    @functools.partial(jax.value_and_grad, has_aux=True)
    def value_and_grad_fn(params, state, rng_per_example, inputs):
      (loss, noise), state = simple_net.apply(
          params, state, rng_per_example, inputs)
      metrics = typing.Metrics(
          per_example={'noise': noise},
      )
      return loss, (state, metrics)

    with self.subTest('vectorized'):
      grad_fn = grad_clipping.value_and_clipped_grad_vectorized(
          value_and_grad_fn, clipping_fn=self.no_clip)
      (_, (_, metrics)), _ = self.variant(grad_fn)(*self.grad_fn_args)

      # Noise should be different across all samples.
      for i in range(1, NUM_SAMPLES):
        self.assertTrue(
            jnp.any(metrics.per_example['noise'][0]
                    != metrics.per_example['noise'][i]))

    with self.subTest('loop'):
      grad_fn = grad_clipping.value_and_clipped_grad_loop(
          value_and_grad_fn, clipping_fn=self.no_clip)
      (_, (_, metrics)), _ = self.variant(grad_fn)(*self.grad_fn_args)

      # Noise should be different across all samples.
      for i in range(1, NUM_SAMPLES):
        self.assertTrue(
            jnp.any(metrics.per_example['noise'][0]
                    != metrics.per_example['noise'][i]))


if __name__ == '__main__':
  absltest.main()
