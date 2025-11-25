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

from collections.abc import Mapping
import functools
from typing import Protocol
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import grad_clipping_utils
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import typing
import numpy as np
import optax

jax.config.update('jax_threefry_partitionable', False)


def _tree_dot(w: chex.ArrayTree, x: chex.ArrayTree) -> chex.Array:
  """Returns scalar (dot) product of two compatible array trees."""
  per_node_products = jax.tree_util.tree_map(lambda a, b: jnp.sum(a * b), w, x)
  flat_products, _ = jax.tree_util.tree_flatten(per_node_products)
  return sum(flat_products)


_RNG_SCALE = 1.0e7
_LOSS_WEIGHT = 0.7


class GradientComputerFn(Protocol):

  def __call__(
      self,
      *,
      clipping_norm: float | None,
      rescale_to_unit_norm: bool,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      noise_multiplier: float | None = None,
  ) -> gradients.GradientComputer:
    """Returns a constructed `gradients.GradientComputer` using the args."""


def _make_dpsgd_gradient_computer(
    clipping_norm: float | None,
    rescale_to_unit_norm: bool,
    per_example_grad_method: grad_clipping.PerExampleGradMethod,
    noise_multiplier: float | None = None,
) -> gradients.GradientComputer:
  return gradients.DpsgdGradientComputer(
      clipping_norm=clipping_norm,
      noise_multiplier=noise_multiplier,
      rescale_to_unit_norm=rescale_to_unit_norm,
      per_example_grad_method=per_example_grad_method,
  )


def _make_dpftrl_gradient_computer(
    correlation_matrix: jax.Array,
    *,
    clipping_norm: float | None,
    rescale_to_unit_norm: bool,
    per_example_grad_method: grad_clipping.PerExampleGradMethod,
    noise_multiplier: float | None = None,
    correlation_unroll: int | bool | None = False,
) -> gradients.GradientComputer:
  return gradients.DpftrlGradientComputer(
      clipping_norm=clipping_norm,
      noise_multiplier=noise_multiplier,
      rescale_to_unit_norm=rescale_to_unit_norm,
      per_example_grad_method=per_example_grad_method,
      correlation_matrix=correlation_matrix,
      correlation_unroll=correlation_unroll,
  )


def _identity_dpftrl_gradient_computer_fn(num_steps: int) -> GradientComputerFn:
  return lambda *args, **kwargs: _make_dpftrl_gradient_computer(
      jnp.eye(num_steps), *args, **kwargs
  )


def _allow_any_matrix_dpftrl(testcase):
  """Wraps a testcase to allow matrix for `gradients.DpftrlGradientcomputer`."""
  empty_func = lambda *args, **kwargs: None
  ignore_fn = functools.partial(mock.patch.object, gradients, new=empty_func)
  ignore_square = ignore_fn(attribute='check_square')
  ignore_tril = ignore_fn(attribute='check_lower_triangular')
  return ignore_tril(ignore_square(testcase))


def _cross_computers(
    *testcases,
    grad_computer_fns: Mapping[str, GradientComputerFn],
):
  """Expands testcases, adding each `grad_computer_fn` to each testcase.

  Essentially, we compute a cross product between the `testcases` and each
  desired gradient computer in `grad_computer_fns`. This creates n * m new
  testcases where n = len(`testcases`) and m = len(`grad_computer_fns`).

  A new argument is appended to each testcase representing the
  `gradient_computer_fn`. The key in `grad_computer_fns` is used to rename
  each testcase.

  Args:
    *testcases: the named testcases whose first arg must be a string
      representing the testcase name, and the remaining arguments specify the
      testcase.
    grad_computer_fns: a dict where the key represents a unique shortname to
      append to each testcase name and whose value is a `GradientComputerFn`
      that will be appended as the final arg in the testcase.

  Returns:
    Cross product of testcases and gradient computers in
    `grad_computer_fns`.
  """
  output_tests = []
  for shortname, grad_computer_fn in grad_computer_fns.items():
    suffix = f"_{shortname.strip().strip('_')}"
    for testcase in testcases:
      name, *args = testcase
      new_test = (name + suffix, *args, grad_computer_fn)
      output_tests.append(new_test)
  return output_tests


class GradientsTest(chex.TestCase):

  @parameterized.named_parameters(
      ('dpsgd', _make_dpsgd_gradient_computer),
      ('dpftrl_identity', _identity_dpftrl_gradient_computer_fn(1)),
  )
  def test_clean_gradients(self, make_gradient_computer_fn):
    gradient_computer = make_gradient_computer_fn(
        clipping_norm=None,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
    )

    inputs = jnp.array([3.0, 4.0])
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    avg_grads = gradient_computer.clean_gradients(
        loss_fn=testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
    )

    # Gradients are expected to be _LOSS_WEIGHT * inputs, as arranged by
    # `self._testing_loss`.
    chex.assert_trees_all_close(
        jax.tree_util.tree_map(
            lambda t: jnp.mean(t, axis=0) * _LOSS_WEIGHT, inputs
        ),
        avg_grads['w_inputs'],
    )
    chex.assert_trees_all_close(
        jax.tree_util.tree_map(lambda t: t * _LOSS_WEIGHT, network_state),
        avg_grads['w_network_state'],
    )

  @parameterized.named_parameters(
      _cross_computers(
          ('no_clipping', None, False),
          ('vacuous_clipping_looped', 1.0e10, grad_clipping.UNROLLED),
          ('vacuous_clipping_vectorised', 1.0e10, grad_clipping.VECTORIZED),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_non_clipped_gradients(
      self,
      clipping_norm: float,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        per_example_grad_method=per_example_grad_method,
    )

    inputs = jnp.array([[3.0, 4.0], [5.0, 7.0]])
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        loss_fn=testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
        state_acc_strategies=grad_clipping_utils.Average(),
    )

    # Gradients are expected to be _LOSS_WEIGHT * inputs, as arranged by
    # `self._testing_loss`.
    chex.assert_trees_all_close(
        jax.tree_util.tree_map(
            lambda t: jnp.mean(t, axis=0) * _LOSS_WEIGHT, inputs
        ),
        avg_grads['w_inputs'],
    )
    chex.assert_trees_all_close(
        jax.tree_util.tree_map(lambda t: t * _LOSS_WEIGHT, network_state),
        avg_grads['w_network_state'],
    )

  @parameterized.named_parameters(
      _cross_computers(
          ('1e-5', 1.0e-5, True),
          ('3e-2', 3.0e-2, False),
          ('1', 1.0, True),
          ('20', 20.0, False),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_clipped_gradients_looped_equal_vectorised(
      self,
      clipping_norm: float,
      rescale_to_unit_norm: bool,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=grad_clipping.UNROLLED,
    )

    gradient_computer_v = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=grad_clipping.VECTORIZED,
    )

    inputs = jnp.array([[3.0, 4.0], [5.0, 7.0]])
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    testing_loss = functools.partial(self._testing_loss, include_rngs=True)
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        loss_fn=testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
        state_acc_strategies=grad_clipping_utils.Average(),
    )
    _, avg_grads_v = gradient_computer_v.loss_and_clipped_gradients(
        loss_fn=testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
        state_acc_strategies=grad_clipping_utils.Average(),
    )

    chex.assert_trees_all_close(avg_grads, avg_grads_v)

  @parameterized.named_parameters(
      _cross_computers(
          ('noscale_looped', False, grad_clipping.UNROLLED),
          ('noscale_vectorised', False, grad_clipping.VECTORIZED),
          ('rescale_looped', True, grad_clipping.UNROLLED),
          ('rescale_vectorised', True, grad_clipping.VECTORIZED),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_tightly_clipped_correctly_normalised(
      self,
      rescale_to_unit_norm: bool,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      gradient_computer_fn: GradientComputerFn,
  ):
    clipping_norm = 1.0e-2
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=per_example_grad_method,
    )

    inputs = jnp.array([[3.0, 4.0, 1.0], [5.0, 7.0, 2.0]])
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    batch_size = inputs.shape[0]

    clean_grads_per_example = [
        gradient_computer.clean_gradients(
            loss_fn=self._testing_loss,
            params=params,
            network_state=network_state,
            rng_per_local_microbatch=rng_per_batch,
            inputs=inputs[i : i + 1],
        )
        for i in range(batch_size)
    ]
    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        loss_fn=self._testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
        state_acc_strategies=grad_clipping_utils.Average(),
    )

    # Assuming that the clipping will be effective for each example,
    # we expect each example's tree of gradients to be normalised to
    # `clipping_norm`. These are then averaged across examples.
    clean_grad_norms = [
        optax.global_norm(clean_grads)
        for clean_grads in clean_grads_per_example
    ]
    normalised_grads = [
        jax.tree_util.tree_map(
            lambda x, i=i: x / clean_grad_norms[i], clean_grads_per_example[i]
        )
        for i in range(batch_size)
    ]
    expected_avg_grads = jax.tree_util.tree_map(
        lambda *x: sum(x) / batch_size, *normalised_grads
    )
    if not rescale_to_unit_norm:
      expected_avg_grads = jax.tree_util.tree_map(
          lambda x: x * clipping_norm, expected_avg_grads
      )
    chex.assert_trees_all_close(expected_avg_grads, avg_grads)

  @parameterized.named_parameters(
      _cross_computers(
          ('no_clipping', None, grad_clipping.UNROLLED),
          ('clipping', 3.0, grad_clipping.UNROLLED),
          ('clipping_vectorised', 3.0, grad_clipping.VECTORIZED),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_batch_size_1(
      self,
      clipping_norm: float,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        per_example_grad_method=per_example_grad_method,
    )

    # Test that a single example gives the same (averaged) gradients as
    # a batch of several identical copies of it.
    inputs = jnp.array([[3.0, 8.0, 5.0]])
    inputs_dup = jnp.array([inputs] * 3)
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    _, avg_grads = gradient_computer.loss_and_clipped_gradients(
        loss_fn=self._testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs,
        state_acc_strategies=grad_clipping_utils.Average(),
    )
    _, avg_grads_dup = gradient_computer.loss_and_clipped_gradients(
        loss_fn=self._testing_loss,
        params=params,
        network_state=network_state,
        rng_per_local_microbatch=rng_per_batch,
        inputs=inputs_dup,
        state_acc_strategies=grad_clipping_utils.Average(),
    )

    for key in ('w_inputs', 'w_network_state'):
      chex.assert_trees_all_close(
          avg_grads[key], avg_grads_dup[key], atol=1.0e-6
      )

  @parameterized.named_parameters(
      _cross_computers(
          ('no_clipping', None, grad_clipping.UNROLLED, 5),
          ('no_clipping_batch_size_1', None, grad_clipping.UNROLLED, 1),
          ('vacuous_clipping_looped', 1.0, grad_clipping.UNROLLED, 5),
          (
              'vacuous_clipping_looped_batch_size_1',
              1.0,
              grad_clipping.UNROLLED,
              1,
          ),
          ('vacuous_clipping_vectorised', 1.0, grad_clipping.VECTORIZED, 5),
          (
              'vacuous_clipping_vectorised_batch_size_1',
              1.0,
              grad_clipping.VECTORIZED,
              1,
          ),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_aux_aggregation(
      self,
      clipping_norm: float,
      per_example_grad_method: grad_clipping.PerExampleGradMethod,
      batch_size: int,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        per_example_grad_method=per_example_grad_method,
    )

    inputs = jnp.array(
        [[3.0, 4.0], [5.0, 7.0], [2.0, -1.0], [1.0, 0.0], [3.0, 1.0]]
    )
    inputs = inputs[:batch_size]
    network_state = {'k': jnp.array(5.0)}
    params = self._params_for_testing_loss(inputs, network_state)
    rng_per_batch = jax.random.PRNGKey(54)

    ((loss, (new_network_state, metrics)), unused_grads) = (
        gradient_computer.loss_and_clipped_gradients(
            loss_fn=self._testing_loss,
            params=params,
            network_state=network_state,
            rng_per_local_microbatch=rng_per_batch,
            inputs=inputs,
            state_acc_strategies=grad_clipping_utils.Average(),
        )
    )

    chex.assert_trees_all_close(network_state, new_network_state)
    # Averaged.
    chex.assert_shape(loss, ())
    chex.assert_shape(metrics.scalars_avg.get('aggregate'), (3,))
    # Stacked, over all devices.
    if clipping_norm:
      chex.assert_shape(metrics.per_example.get('grad_norm'), (batch_size,))
    chex.assert_shape(metrics.per_example.get('loss'), (batch_size,))
    chex.assert_shape(metrics.per_example.get('other'), (batch_size, 3, 2))

  @parameterized.named_parameters(
      _cross_computers(
          ('None', None),
          ('3', 3.0),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_adding_zero_noise(
      self,
      clipping_norm: float,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=0.0,
    )
    grads = {
        'a': jnp.array(3.0),
        'b': [jnp.array([4.0, 5.0]), jnp.array([6.0])],
    }
    rng_per_batch = jax.random.PRNGKey(54)
    noisy_grads, std, unused_noise_state = gradient_computer.add_noise_to_grads(
        grads=grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=jnp.array(8),
        noise_state=gradient_computer.init_noise_state(grads),
    )

    chex.assert_trees_all_close(grads, noisy_grads)
    self.assertEqual(std, 0.0)

  @parameterized.named_parameters(
      _cross_computers(
          ('False', False),
          ('True', True),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_cannot_add_noise_without_clipping(
      self,
      rescale_to_unit_norm: bool,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=None,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=grad_clipping.UNROLLED,
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
          noise_state=gradient_computer.init_noise_state(grads),
      )

  @parameterized.named_parameters(
      _cross_computers(
          ('clip=0.0_nm=0.0', 0.0, False, 0.0, 4, 0.0),
          ('clip=0.1_nm=0.0', 0.1, False, 0.0, 4, 0.0),
          ('clip=0.1_nm=3.0', 0.1, False, 3.0, 4, 0.075),
          ('clip=0.1_nm=3.0,rescaled', 0.1, True, 3.0, 4, 0.75),
          ('clip=0.10_nm=5.0', 10.0, False, 5.0, 4, 12.5),
          ('clip=0.10_nm=5.0,rescaled', 10.0, True, 5.0, 4, 1.25),
          grad_computer_fns={
              'dpsgd': _make_dpsgd_gradient_computer,
              'dpftrl': _identity_dpftrl_gradient_computer_fn(2),
          },
      )
  )
  def test_adding_noise(  # pylint: disable=too-many-positional-arguments
      self,
      clipping_norm: float,
      rescale_to_unit_norm: bool,
      noise_multiplier: float,
      total_batch_size: int,
      expected_std: float,
      gradient_computer_fn: GradientComputerFn,
  ):
    gradient_computer = gradient_computer_fn(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=rescale_to_unit_norm,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=noise_multiplier,
    )

    grads = jnp.zeros((1_000_000,))
    rng_per_batch = jax.random.PRNGKey(54)
    noisy_grads, std, unused_noise_state = gradient_computer.add_noise_to_grads(
        grads,
        rng_per_batch,
        total_batch_size=jnp.array(total_batch_size),
        noise_state=gradient_computer.init_noise_state(grads),
    )

    np.testing.assert_approx_equal(expected_std, std)
    np.testing.assert_approx_equal(
        np.mean(noisy_grads**2), std**2, significant=2
    )

  def _testing_loss(  # pylint: disable=too-many-positional-arguments
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
    inputs_loss = jax.tree_util.tree_map(
        sum_to_mean, _tree_dot(params['w_inputs'], inputs)
    )
    weights_loss = _tree_dot(params['w_network_state'], network_state)
    rng_loss = _tree_dot(
        params['w_rng_per_example'], rng_per_example / _RNG_SCALE
    )
    loss = (inputs_loss + weights_loss + include_rngs * rng_loss) * _LOSS_WEIGHT
    metrics = typing.Metrics(
        scalars_avg={'aggregate': jnp.array([1.0, 2.0, 3.0])},
        per_example={
            'loss': loss * jnp.ones((batch_size,)),
            'other': jnp.ones((batch_size, 3, 2)),
        },
    )
    return loss, (network_state, metrics)

  def _params_for_testing_loss(self, inputs, network_state):
    return {
        'w_inputs': jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x[0]), inputs
        ),
        'w_network_state': jax.tree_util.tree_map(
            jnp.zeros_like, network_state
        ),
        'w_rng_per_example': jnp.zeros(jax.random.PRNGKey(0).shape),
    }


class CorrelatedGradientsTest(chex.TestCase):

  @parameterized.parameters((1,), (3,), (5,))
  def test_error_too_few_rows(self, matrix_size):
    gradient_computer = _make_dpftrl_gradient_computer(
        jnp.eye(matrix_size),
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=0.0,
    )
    grads = {
        'a': jnp.array(3.0),
        'b': [jnp.array([4.0, 5.0]), jnp.array([6.0])],
    }
    rng_per_batch = jax.random.PRNGKey(54)
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        grads=grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=jnp.array(8),
    )
    noise_state = gradient_computer.init_noise_state(grads)
    for _ in range(matrix_size):
      _, _, noise_state = grad_fn(
          noise_state=noise_state,
      )

    with self.assertRaises(
        ValueError,
        msg=(
            'DPFTRL Gradient Computer should raise when correlation matrix is'
            ' exhausted'
        ),
    ):
      _ = grad_fn(
          noise_state=noise_state,
      )

  @_allow_any_matrix_dpftrl
  def test_noise_cancels(self):
    correlation_matrix = jnp.asarray([[1.0, -1.0]])
    gradient_computer = _make_dpftrl_gradient_computer(
        correlation_matrix,
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=1.0,
    )
    grads = {
        'a': jnp.array(3.0),
        'b': [jnp.array([4.0, 5.0]), jnp.array([6.0])],
    }
    rng_per_batch = jax.random.PRNGKey(54)
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=jnp.array(8),
    )
    noise_state = gradient_computer.init_noise_state(grads)
    noised_grads, _, _ = grad_fn(grads=grads, noise_state=noise_state)
    chex.assert_trees_all_close(grads, noised_grads)

  def test_noise_iterates_rows_diag(self):
    num_stds = 4
    stds = jnp.power(10.0, jnp.arange(-num_stds, 0))
    correlation_matrix = jnp.eye(num_stds, dtype=jnp.float64) * stds

    gradient_computer = _make_dpftrl_gradient_computer(
        correlation_matrix,
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=1.0,
    )
    grads = jnp.ones((10_000,), dtype=jnp.float64)
    rng_per_batch = jax.random.PRNGKey(12)
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        grads=grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=jnp.array(1),
    )

    noise_state = gradient_computer.init_noise_state(grads)
    for std in stds:
      noises, _, noise_state = grad_fn(noise_state=noise_state)
      np.testing.assert_allclose(jnp.std(noises), std, rtol=5e-2)

  def test_noise_iterates_rows_lower_tri(self):
    num_stds = 4
    stds = jnp.power(10.0, jnp.arange(-num_stds, 0))
    correlation_matrix = (
        jnp.tril(jnp.ones((num_stds, num_stds), dtype=jnp.float64)) * stds
    )

    gradient_computer = _make_dpftrl_gradient_computer(
        correlation_matrix,
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=1.0,
    )
    grads = jnp.ones((10_000,), dtype=jnp.float64)
    rng_per_batch = jax.random.PRNGKey(54)
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        grads=grads,
        total_batch_size=jnp.array(1),
    )

    noise_state = gradient_computer.init_noise_state(grads)
    for i, row in enumerate(correlation_matrix):
      noises, _, noise_state = grad_fn(
          noise_state=noise_state, rng_per_batch=rng_per_batch
      )
      rng_per_batch = jax.random.fold_in(rng_per_batch, i)
      np.testing.assert_allclose(
          jnp.std(noises), jnp.linalg.norm(row, 2), rtol=2e-1
      )

  def test_past_noise_deterministic(self):
    # Matrix with first column of 1's and rest of 0's.
    num_iters = 10
    correlation_matrix = jnp.vstack(
        [jnp.ones(10)] + [jnp.zeros(10)] * (num_iters - 1)
    ).T

    gradient_computer = _make_dpftrl_gradient_computer(
        correlation_matrix,
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=1.0,
    )
    grads = jnp.ones((10,))
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        grads=grads,
        total_batch_size=jnp.array(8),
    )
    rng_per_batch = jax.random.PRNGKey(54)
    init_state = gradient_computer.init_noise_state(grads)
    initial_noise, _, next_state = grad_fn(
        noise_state=init_state, rng_per_batch=rng_per_batch
    )
    for i in range(num_iters - 1):
      rng_per_batch = jax.random.fold_in(rng_per_batch, i)
      next_iter_noise, _, next_state = grad_fn(
          noise_state=next_state, rng_per_batch=rng_per_batch
      )
      chex.assert_trees_all_close(
          initial_noise,
          next_iter_noise,
          custom_message=(
              f'Iteration `{i + 1}` failed to regenerate deterministic noise.'
          ),
      )

  @parameterized.parameters(
      (1.0, 1, 1), (3.0, 1, 1), (3.0, 8, 1), (3.0, 8, 0.1)
  )
  @_allow_any_matrix_dpftrl
  def test_std_calculation_correct(self, nm, batch_size, clipping_norm):
    batch_size = jnp.array(batch_size)
    stds = jnp.arange(10).astype(jnp.float64)
    correlation_matrix = jnp.asarray([stds])
    gradient_computer = _make_dpftrl_gradient_computer(
        correlation_matrix,
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=nm,
    )
    grads = jnp.ones((100_000,))
    rng_per_batch = jax.random.PRNGKey(54)
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        rng_per_batch=rng_per_batch,
        total_batch_size=batch_size,
    )
    noise_state = gradient_computer.init_noise_state(grads)
    _, std, _ = grad_fn(grads=grads, noise_state=noise_state)
    expected_std = jnp.linalg.norm(stds / batch_size, 2) * nm * clipping_norm
    self.assertAlmostEqual(std, expected_std, 2)

  @parameterized.named_parameters(
      ('non_matrix', lambda: jnp.zeros((10, 10, 10))),
      ('non_square', lambda: jnp.zeros((10, 11))),
      ('non_lower_triangular', lambda: jnp.ones((10, 10))),
  )
  def test_invalid_matrices_raises(self, correlation_matrix_fn):
    with self.assertRaises(
        ValueError, msg='Invalid correlation matrix should raise.'
    ):
      _make_dpftrl_gradient_computer(
          correlation_matrix_fn(),
          clipping_norm=1.0,
          rescale_to_unit_norm=False,
          per_example_grad_method=grad_clipping.UNROLLED,
          noise_multiplier=1.0,
      )

  @parameterized.parameters([0, 1], [1, 2], [2, 3])
  def test_diff_seeds_diff_noise(self, seed1, seed2):
    gradient_computer_fn = _identity_dpftrl_gradient_computer_fn(1)
    gradient_computer = gradient_computer_fn(
        clipping_norm=1.0,
        rescale_to_unit_norm=False,
        per_example_grad_method=grad_clipping.UNROLLED,
        noise_multiplier=1.0,
    )

    grads = jnp.zeros((1,))
    grad_fn = functools.partial(
        gradient_computer.add_noise_to_grads,
        grads=grads,
        total_batch_size=jnp.array(1),
    )
    rng_per_batch1 = jax.random.PRNGKey(seed1)
    rng_per_batch2 = jax.random.PRNGKey(seed2)
    state = gradient_computer.init_noise_state(grads)

    noise1, _, _ = grad_fn(noise_state=state, rng_per_batch=rng_per_batch1)
    noise2, _, _ = grad_fn(noise_state=state, rng_per_batch=rng_per_batch2)
    self.assertNotEqual(
        noise1,
        noise2,
        msg=(
            f'Different seeds generated the same noise: {noise1}, and {noise2}.'
        ),
    )


if __name__ == '__main__':
  absltest.main()
