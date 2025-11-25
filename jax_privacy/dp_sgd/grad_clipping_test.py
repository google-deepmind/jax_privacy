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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import grad_clipping_utils
from jax_privacy.dp_sgd import typing
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


def _mask_out_all_others(i: int, x: chex.Array) -> chex.Array:
  mask = jnp.arange(x.shape[0]) == i
  expanded_mask = jnp.expand_dims(mask, axis=list(range(x.ndim))[1:])
  return expanded_mask * x


_PER_EXAMPLE_GRAD_METHODS = [
    ('vectorised_vmap_of_reshape', grad_clipping.VECTORIZED),
    ('unrolled', grad_clipping.UNROLLED),
    (
        'mask_unrolled',
        grad_clipping.unrolled_grad_method(
            select_example_fn=_mask_out_all_others,
            select_example_metric_fn=lambda i, x: x[i],
        ),
    ),
    (
        'vectorized_reshape_then_vmap',
        grad_clipping._value_and_clipped_grad_reshape_then_vmap,
    ),
]


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
    images, labels, loss_weights = inputs
    batch_size = len(images)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad_norms = []

    # compute one clipped gradient at a time
    for i in range(batch_size):
      # Forward function expects a batch dimension.
      input_i = (
          jnp.expand_dims(images[i], 0),
          jnp.expand_dims(labels[i], 0),
          jnp.expand_dims(loss_weights[i], 0),
      )
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

    images = jax.random.normal(next(rng_seq), shape=(NUM_SAMPLES, *INPUT_SHAPE))
    labels = jax.random.randint(
        next(rng_seq), shape=[NUM_SAMPLES], minval=0, maxval=NUM_CLASSES)
    loss_weights = jnp.ones([NUM_SAMPLES])

    self.net = hk.transform_with_state(
        functools.partial(model_fn, num_classes=NUM_CLASSES))
    self.params, self.network_state = self.net.init(next(rng_seq), images)

    self.inputs = (images, labels, loss_weights)
    self.rng_per_batch = next(rng_seq)
    self.rng_per_example = jax.random.fold_in(self.rng_per_batch, 1)

    self.tol = {'rtol': 1e-6, 'atol': 1e-6}

  def forward(self, params, state, rng_per_example, inputs):
    """Forward pass for the model."""
    images, labels, loss_weights = inputs
    logits, state = self.net.apply(params, state, rng_per_example, images)
    weighted_labels_one_hot = (
        hk.one_hot(labels, NUM_CLASSES) * jnp.expand_dims(loss_weights, axis=1))

    loss_vector = optax.softmax_cross_entropy(logits, weighted_labels_one_hot)
    average_loss = jnp.sum(loss_vector) / jnp.sum(loss_weights)
    ones_per_example = jnp.ones([images.shape[0]])
    # Expect integer to be averaged to float32.
    one_to_average = jnp.ones([], dtype=jnp.int32)

    metrics = typing.Metrics(
        per_example={'loss': loss_vector, 'a': ones_per_example},
        scalars_avg={'loss': average_loss, 'b': one_to_average},
    )
    return average_loss, (state, metrics)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(_PER_EXAMPLE_GRAD_METHODS)
  def test_clipped_gradients(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    value_and_grad_fn = jax.value_and_grad(self.forward, has_aux=True)
    clipping_fn = grad_clipping.global_clipping(clipping_norm=MAX_NORM)

    grad_fn = per_example_grad_method(
        value_and_grad_fn,
        clipping_fn,
        state_acc_strategies=grad_clipping_utils.Reject(),
    )

    grad_fn_naive = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=MAX_NORM,
    )

    (loss_actual, aux_actual), grad_actual = self.variant(grad_fn)(
        self.params, self.network_state, self.rng_per_example, self.inputs)
    (loss_expected, aux_expected), grad_expected = self.variant(grad_fn_naive)(
        self.params, self.network_state, self.rng_per_example, self.inputs)

    chex.assert_trees_all_close(loss_actual, loss_expected, **self.tol)
    chex.assert_trees_all_close(aux_actual, aux_expected, **self.tol)
    chex.assert_trees_all_close(grad_actual, grad_expected, **self.tol)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(_PER_EXAMPLE_GRAD_METHODS)
  def test_gradients_vectorized_and_loop_match_using_batch_rng(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    value_and_grad_fn = jax.value_and_grad(self.forward, has_aux=True)
    clipping_fn = lambda grads: (grads, optax.global_norm(grads))

    grad_fn = per_example_grad_method(
        value_and_grad_fn,
        clipping_fn=clipping_fn,
        state_acc_strategies=grad_clipping_utils.Reject(),
    )

    grad_fn_naive = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=float('inf'),
    )

    (loss_actual, aux_actual), grad_actual = self.variant(grad_fn)(
        self.params, self.network_state, self.rng_per_example, self.inputs)
    (loss_expected, aux_expected), grad_expected = self.variant(grad_fn_naive)(
        self.params, self.network_state, self.rng_per_example, self.inputs)

    chex.assert_trees_all_close(loss_actual, loss_expected, **self.tol)
    chex.assert_trees_all_close(aux_actual, aux_expected, **self.tol)
    chex.assert_trees_all_close(grad_actual, grad_expected, **self.tol)


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
  @parameterized.named_parameters(_PER_EXAMPLE_GRAD_METHODS)
  def test_per_sample_rng_produces_different_random_numbers(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):

    @functools.partial(jax.value_and_grad, has_aux=True)
    def value_and_grad_fn(params, state, rng_per_example, inputs):
      (loss, noise), state = simple_net.apply(
          params, state, rng_per_example, inputs)
      metrics = typing.Metrics(
          per_example={'noise': noise},
      )
      return loss, (state, metrics)

    with self.subTest('loop'):
      grad_fn = per_example_grad_method(
          value_and_grad_fn,
          clipping_fn=self.no_clip,
          state_acc_strategies=grad_clipping_utils.Reject(),
      )
      (_, (_, metrics)), _ = self.variant(grad_fn)(*self.grad_fn_args)

      # Noise should be different across all samples.
      for i in range(1, NUM_SAMPLES):
        self.assertTrue(
            jnp.any(metrics.per_example['noise'][0]
                    != metrics.per_example['noise'][i]))


@parameterized.named_parameters(_PER_EXAMPLE_GRAD_METHODS)
class TestNetworkStateAccumulation(chex.TestCase):
  """Tests the various strategies for accumulating network state."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.no_clip = lambda grads: (grads, optax.global_norm(grads))

  def _value_and_grad_fn(
      self,
      net: hk.TransformedWithState,
      *,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      state_acc_strategies: grad_clipping_utils.StateAccumulationStrategyTree,
  ) -> typing.ValueAndGradFn:
    """Returns a value_and_grad_fn given a state accumulation strategy."""

    @functools.partial(jax.value_and_grad, has_aux=True)
    def value_and_grad_fn(params, state, rng_per_example, inputs):
      loss, state = net.apply(params, state, rng_per_example, inputs)
      return loss, (state, typing.Metrics())

    return per_example_grad_method(
        value_and_grad_fn,
        clipping_fn=self.no_clip,
        state_acc_strategies=state_acc_strategies,
    )

  def test_no_state_passes(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    @hk.transform_with_state
    def net(x):
      return jnp.sum(x)

    value_and_grad_fn = self._value_and_grad_fn(
        net,
        per_example_grad_method=per_example_grad_method,
        state_acc_strategies=grad_clipping_utils.Reject(),
    )
    params, state = net.init(self.rng, jnp.zeros(shape=(7,)))
    (unused_loss, (state, unused_metrics)), params = value_and_grad_fn(
        params, state, self.rng, jnp.zeros(shape=(7,)))
    del params

    chex.assert_trees_all_close({}, state)

  def test_state_triggers_error(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    @hk.transform_with_state
    def net(x):
      hk.set_state('s', jnp.array([3.]))
      return jnp.sum(x)

    with self.assertRaises(ValueError) as context:
      value_and_grad_fn = self._value_and_grad_fn(
          net,
          per_example_grad_method=per_example_grad_method,
          state_acc_strategies=grad_clipping_utils.Reject(),
      )
      params, state = net.init(self.rng, jnp.zeros(shape=(7,)))
      value_and_grad_fn(params, state, self.rng, jnp.zeros(shape=(7,)))
    self.assertContainsSubset('Unhandled network state', str(context.exception))

  def test_state_with_average(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):

    @hk.transform_with_state
    def net(x):
      # For testing purposes, the state just echoes the inputs.
      with hk.experimental.name_scope('a'):
        hk.set_state('c', jnp.sum(x, axis=0))
      return jnp.sum(x)

    value_and_grad_fn = self._value_and_grad_fn(
        net,
        per_example_grad_method=per_example_grad_method,
        state_acc_strategies=grad_clipping_utils.Average(),
    )
    params, state = net.init(self.rng, jnp.zeros(shape=(7,)))
    (unused_loss, (state, unused_metrics)), params = value_and_grad_fn(
        params, state, self.rng, jnp.arange(7, dtype=jnp.float32))
    del params

    # Vectorised state was [0, 1, ..., 6].
    # Expect this to be averaged.
    chex.assert_trees_all_close({'a': {'c': 3.}}, state)

  def test_state_with_sum(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):

    @hk.transform_with_state
    def net(x):
      # For testing purposes, the state is a running sum of the inputs.
      with hk.experimental.name_scope('a'):
        c = hk.get_state(
            'c', shape=x.shape[1:], dtype=jnp.float32, init=jnp.zeros)
        hk.set_state('c', c + jnp.sum(x, axis=0))
      return jnp.sum(x)

    value_and_grad_fn = self._value_and_grad_fn(
        net,
        per_example_grad_method=per_example_grad_method,
        state_acc_strategies=grad_clipping_utils.Sum(),
    )
    params, state = net.init(self.rng, jnp.zeros(shape=(5,)))
    (unused_loss, (state, unused_metrics)), params = value_and_grad_fn(
        params, state, self.rng, jnp.arange(5, dtype=jnp.float32))

    # Vectorised state was [0, 1, 2, 3, 4].
    # Expect this to be summed.
    chex.assert_trees_all_close({'a': {'c': 10.}}, state)

    (unused_loss, (state, unused_metrics)), params = value_and_grad_fn(
        params, state, self.rng, 2. * jnp.arange(5, dtype=jnp.float32))
    del params

    # Vectorised state was now [10, 12, 14, 16, 18].
    # Expect this to be summed relative to the previous value 10.
    chex.assert_trees_all_close({'a': {'c': 30.}}, state)

  def test_incomplete_strategy_triggers_error(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    @hk.transform_with_state
    def net(x):
      with hk.experimental.name_scope('a'):
        hk.set_state('s', jnp.array([3.]))
        hk.set_state('t', jnp.array([4.]))
      return jnp.sum(x)

    with self.assertRaises(ValueError) as context:
      value_and_grad_fn = self._value_and_grad_fn(
          net,
          per_example_grad_method=per_example_grad_method,
          state_acc_strategies={
              'a': {
                  's': grad_clipping_utils.Average(),
                  't': grad_clipping_utils.Reject(),
              },
          },
      )
      params, state = net.init(self.rng, jnp.zeros(shape=(7,)))
      value_and_grad_fn(params, state, self.rng, jnp.zeros(shape=(7,)))
    self.assertContainsSubset('Unhandled network state', str(context.exception))

  def test_mixed_strategies(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod):
    @hk.transform_with_state
    def net(x):
      m = jnp.sum(x)  # 0 in init; 1 in apply.
      with hk.experimental.name_scope('a'):
        hk.set_state('s', m*jnp.array([3.]))
        hk.set_state('t', m*jnp.array([4.]))
      with hk.experimental.name_scope('b'):
        hk.set_state('u', m*jnp.array([6.]))
        hk.set_state('v', m*jnp.array([7.]))
      return jnp.sum(x)

    value_and_grad_fn = self._value_and_grad_fn(
        net,
        per_example_grad_method=per_example_grad_method,
        state_acc_strategies={
            'a': {
                's': grad_clipping_utils.Average(),
                't': grad_clipping_utils.Sum(),
            },
            'b': grad_clipping_utils.Average(),
        },
    )
    params, state = net.init(self.rng, jnp.zeros(shape=(11,)))
    print(state)
    (unused_loss, (state, unused_metrics)), params = value_and_grad_fn(
        params, state, self.rng, jnp.ones(shape=(11,)))
    del params

    # Expect "a.t" to have been summed, and the rest to have been averaged.
    chex.assert_trees_all_close({
        'a': {'s': 3., 't': 44.},
        'b': {'u': 6., 'v': 7.},
    }, state)


if __name__ == '__main__':
  absltest.main()
