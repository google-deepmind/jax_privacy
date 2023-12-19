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

"""Tests for `gradients.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import grad_clipping_utils
from jax_privacy.src.dp_sgd import gradients
from jax_privacy.src.dp_sgd import typing
import numpy as np
import optax


def _tree_dot(w: chex.ArrayTree, x: chex.ArrayTree) -> chex.Array:
  """Returns scalar (dot) product of two compatible array trees."""
  per_node_products = jax.tree_map(lambda a, b: jnp.sum(a * b), w, x)
  flat_products, _ = jax.tree_util.tree_flatten(per_node_products)
  return sum(flat_products)


_RNG_SCALE = 1.0e7
_LOSS_WEIGHT = 0.7


class GradientsTest(chex.TestCase):

  def _make_gradient_computer(
      self,
      clipping_norm: float | None,
      rescale_to_unit_norm: bool,
      vectorize_grad_clipping: bool,
      noise_multiplier: float | None = None,
  ) -> gradients.GradientComputer:
    return gradients.DpsgdGradientComputer(
        clipping_norm=clipping_norm,
        noise_multiplier=noise_multiplier,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=vectorize_grad_clipping,
    )

  def test_clean_gradients(self):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=None,
        rescale_to_unit_norm=False,
        vectorize_grad_clipping=False,
    )

    inputs = jnp.array([3., 4.])
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    avg_grads = gradient_computer.clean_gradients(
        testing_loss, params, network_state, rng_per_batch, inputs)

    # Gradients are expected to be _LOSS_WEIGHT * inputs, as arranged by
    # `self._testing_loss`.
    chex.assert_trees_all_close(
        jax.tree_map(lambda t: jnp.mean(t, axis=0) * _LOSS_WEIGHT, inputs),
        avg_grads['w_inputs'])
    chex.assert_trees_all_close(
        jax.tree_map(lambda t: t * _LOSS_WEIGHT, network_state),
        avg_grads['w_network_state'])

  @parameterized.named_parameters(
      ('no_clipping', None, False),
      ('vacuous_clipping_looped', 1.e+10, False),
      ('vacuous_clipping_vectorised', 1.e+10, True),
  )
  def test_non_clipped_gradients(self, clipping_norm, vectorize):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        vectorize_grad_clipping=vectorize,
    )

    inputs = jnp.array([[3., 4.], [5., 7.]])
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        testing_loss, params, network_state, rng_per_batch, inputs,
        state_acc_strategies=grad_clipping_utils.Average())

    # Gradients are expected to be _LOSS_WEIGHT * inputs, as arranged by
    # `self._testing_loss`.
    chex.assert_trees_all_close(
        jax.tree_map(lambda t: jnp.mean(t, axis=0) * _LOSS_WEIGHT, inputs),
        avg_grads['w_inputs'])
    chex.assert_trees_all_close(
        jax.tree_map(lambda t: t * _LOSS_WEIGHT, network_state),
        avg_grads['w_network_state'])

  @parameterized.parameters(
      (1.e-5, True),
      (3.e-2, False),
      (1., True),
      (20., False),
  )
  def test_clipped_gradients_looped_equal_vectorised(
      self, clipping_norm, rescale_to_unit_norm):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=False,
    )

    gradient_computer_v = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=True,
    )

    inputs = jnp.array([[3., 4.], [5., 7.]])
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        testing_loss, params, network_state, rng_per_batch, inputs,
        state_acc_strategies=grad_clipping_utils.Average())
    _, avg_grads_v = gradient_computer_v.loss_and_clipped_gradients(
        testing_loss, params, network_state, rng_per_batch, inputs,
        state_acc_strategies=grad_clipping_utils.Average())

    chex.assert_trees_all_close(avg_grads, avg_grads_v)

  @parameterized.named_parameters(
      ('noscale_looped', False, False),
      ('noscale_vectorised', False, True),
      ('rescale_looped', True, False),
      ('rescale_vectorised', True, True),
  )
  def test_tightly_clipped_correctly_normalised(
      self, rescale_to_unit_norm, vectorize):
    clipping_norm = 1.e-2
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=vectorize,
    )

    inputs = jnp.array([[3., 4., 1.], [5., 7., 2.]])
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    batch_size = inputs.shape[0]

    clean_grads_per_example = [
        gradient_computer.clean_gradients(
            self._testing_loss, params, network_state, rng_per_batch,
            inputs[i:i+1]) for i in range(batch_size)]
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        self._testing_loss, params, network_state, rng_per_batch,
        inputs, state_acc_strategies=grad_clipping_utils.Average())

    # Assuming that the clipping will be effective for each example,
    # we expect each example's tree of gradients to be normalised to
    # `clipping_norm`. These are then averaged across examples.
    clean_grad_norms = [
        optax.global_norm(clean_grads)
        for clean_grads in clean_grads_per_example]
    normalised_grads = [
        jax.tree_map(
            lambda x, i=i: x / clean_grad_norms[i],
            clean_grads_per_example[i]
        ) for i in range(batch_size)]
    expected_avg_grads = jax.tree_map(
        lambda *x: sum(x) / batch_size, *normalised_grads)
    if not rescale_to_unit_norm:
      expected_avg_grads = jax.tree_map(
          lambda x: x * clipping_norm, expected_avg_grads)
    chex.assert_trees_all_close(expected_avg_grads, avg_grads)

  @parameterized.named_parameters(
      ('no_clipping', None, False),
      ('clipping', 3., False),
      ('clipping_vectorised', 3., True),
  )
  def test_batch_size_1(self, clipping_norm, vectorize):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        vectorize_grad_clipping=vectorize,
    )

    # Test that a single example gives the same (averaged) gradients as
    # a batch of several identical copies of it.
    inputs = jnp.array([[3., 8., 5.]])
    inputs_dup = jnp.array([inputs] * 3)
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        self._testing_loss, params, network_state, rng_per_batch,
        inputs, state_acc_strategies=grad_clipping_utils.Average())
    _, avg_grads_dup = gradient_computer.loss_and_clipped_gradients(
        self._testing_loss,
        params, network_state, rng_per_batch, inputs_dup,
        state_acc_strategies=grad_clipping_utils.Average())

    for key in ('w_inputs', 'w_network_state'):
      chex.assert_trees_all_close(
          avg_grads[key], avg_grads_dup[key], atol=1.e-6)

  @parameterized.named_parameters(
      ('no_clipping', None, False, 5),
      ('no_clipping_batch_size_1', None, False, 1),
      ('vacuous_clipping_looped', 1., False, 5),
      ('vacuous_clipping_looped_batch_size_1', 1., False, 1),
      ('vacuous_clipping_vectorised', 1., True, 5),
      ('vacuous_clipping_vectorised_batch_size_1', 1., True, 1),
  )
  def test_aux_aggregation(self, clipping_norm, vectorize, batch_size):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        vectorize_grad_clipping=vectorize,
    )

    inputs = jnp.array([[3., 4.], [5., 7.], [2., -1.], [1., 0.], [3., 1.]])
    inputs = inputs[:batch_size]
    network_state = {'k': jnp.array(5.)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    (
        (loss, (new_network_state, metrics)), unused_grads
    ) = gradient_computer.loss_and_clipped_gradients(
        self._testing_loss, params, network_state, rng_per_batch,
        inputs, state_acc_strategies=grad_clipping_utils.Average())

    chex.assert_trees_all_close(network_state, new_network_state)
    # Averaged.
    chex.assert_shape(loss, ())
    chex.assert_shape(metrics.scalars_avg.get('aggregate'), (3,))
    # Stacked, over all devices.
    if clipping_norm:
      chex.assert_shape(metrics.per_example.get('grad_norm'), (batch_size,))
    chex.assert_shape(metrics.per_example.get('loss'), (batch_size,))
    chex.assert_shape(metrics.per_example.get('other'), (batch_size, 3, 2))

  @parameterized.parameters((None,), (3.,))
  def test_adding_zero_noise(self, clipping_norm):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        vectorize_grad_clipping=False,
        noise_multiplier=0.0,
    )
    grads = {'a': jnp.array(3.), 'b': [jnp.array([4., 5.]), jnp.array([6.])]}
    rng_per_batch = jax.random.PRNGKey(54)
    noisy_grads, std, unused_noise_state = gradient_computer.add_noise_to_grads(
        grads=grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=jnp.array(8),
        noise_state=gradient_computer.init_noise(),
    )

    chex.assert_trees_all_close(grads, noisy_grads)
    self.assertEqual(std, 0.0)

  @parameterized.parameters((False,), (True,))
  def test_cannot_add_noise_without_clipping(self, rescale_to_unit_norm):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=None,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=False,
        noise_multiplier=0.2,
    )

    grads = {
        'a': jnp.array(3.0),
        'b': [jnp.array([4.0, 5.0]), jnp.array([6.0])],
    }
    rng_per_batch = jax.random.PRNGKey(54)
    with self.assertRaises(ValueError):
      gradient_computer.add_noise_to_grads(
          grads,
          rng_per_batch,
          total_batch_size=jnp.array(8),
          noise_state=gradient_computer.init_noise(),
      )

  @parameterized.parameters(
      (0.0, False, 0.0, 4, 0.0),
      (0.1, False, 0.0, 4, 0.0),
      (0.1, False, 3.0, 4, 0.075),
      (0.1, True, 3.0, 4, 0.75),
      (10.0, False, 5.0, 4, 12.5),
      (10.0, True, 5.0, 4, 1.25),
  )
  def test_adding_noise(
      self,
      clipping_norm,
      rescale_to_unit_norm,
      noise_multiplier,
      total_batch_size,
      expected_std,
  ):
    gradient_computer = self._make_gradient_computer(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        vectorize_grad_clipping=False,
        noise_multiplier=noise_multiplier,
    )

    grads = jnp.zeros((1_000_000,))
    rng_per_batch = jax.random.PRNGKey(54)
    noisy_grads, std, unused_noise_state = gradient_computer.add_noise_to_grads(
        grads,
        rng_per_batch,
        total_batch_size=jnp.array(total_batch_size),
        noise_state=gradient_computer.init_noise(),
    )

    np.testing.assert_approx_equal(expected_std, std)
    np.testing.assert_approx_equal(
        np.mean(noisy_grads**2), std**2, significant=2
    )

  def _testing_loss(
      self, params, network_state, rng_per_example, inputs, include_rngs=False
  ):
    """Simulates the loss function."""
    # Ensure that random keys have been passed in correctly.
    self.assertEqual((2,), rng_per_example.shape)
    self.assertEqual(jnp.uint32, rng_per_example.dtype)

    # Loss functions MUST be mean-additive.
    batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]
    sum_to_mean = lambda x: x / batch_size

    # Take dot product of params with other inputs, so that the gradients
    # reflect the inputs provided.
    loss = sum([
        jax.tree_map(sum_to_mean, _tree_dot(params['w_inputs'], inputs)),
        _tree_dot(params['w_network_state'], network_state),
        include_rngs * _tree_dot(
            params['w_rng_per_example'], rng_per_example / _RNG_SCALE),
    ]) * _LOSS_WEIGHT
    metrics = typing.Metrics(
        scalars_avg={'aggregate': jnp.array([1., 2., 3.])},
        per_example={
            'loss': loss * jnp.ones((batch_size,)),
            'other': jnp.ones((batch_size, 3, 2)),
        },
    )
    return loss, (network_state, metrics)

  def _params_for_testing_loss(self, inputs, network_state):
    return {
        'w_inputs': jax.tree_map(lambda x: jnp.zeros_like(x[0]), inputs),
        'w_network_state': jax.tree_map(jnp.zeros_like, network_state),
        'w_rng_per_example': jnp.zeros(jax.random.PRNGKey(0).shape),
    }


if __name__ == '__main__':
  absltest.main()
