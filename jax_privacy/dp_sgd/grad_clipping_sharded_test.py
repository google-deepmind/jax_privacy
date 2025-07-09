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
import importlib

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import grad_clipping_utils
from jax_privacy.dp_sgd import typing
import numpy as np
import optax


SPMD_AXIS_NAME = 'clipping_axis'
P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
Mesh = jax.sharding.Mesh


_PER_EXAMPLE_GRAD_METHODS = [
    (
        'vectorised_vmap_of_reshape',
        functools.partial(
            grad_clipping._value_and_clipped_grad_vmap_of_reshape,
            spmd_axis_name=SPMD_AXIS_NAME,
        ),
    ),
    (
        'vectorized_reshape_then_vmap',
        functools.partial(
            grad_clipping._value_and_clipped_grad_reshape_then_vmap,
            spmd_axis_name=SPMD_AXIS_NAME,
            use_shard_alike=False,
        ),
    ),
]

if importlib.util.find_spec('drjax'):
  _PER_EXAMPLE_GRAD_METHODS.append(
      (
          'vectorized_reshape_then_vmap_shard_alike',
          functools.partial(
              grad_clipping._value_and_clipped_grad_reshape_then_vmap,
              spmd_axis_name=SPMD_AXIS_NAME,
              use_shard_alike=True,
          ),
      ),
  )


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
        params, network_state, rng_per_example, inputs
    )
    batch_size = inputs.shape[0]
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    grad_norms = []

    # compute one clipped gradient at a time
    for i in range(batch_size):
      # Forward function expects a batch dimension.
      input_i = jnp.expand_dims(inputs[i], 0)
      grad_i, unused_aux = grad_fn(
          params, network_state, rng_per_example, input_i
      )

      norm_grad_i = jnp.sqrt(
          sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grad_i))
      )

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
    chex.set_n_cpu_devices(8)

    self.model_dim = 5
    self.clipping_norm = 1e-4
    self.batch_size = 8
    # Our model will just be a 1-d tensor.
    self.params = jax.random.normal(
        key=jax.random.key(0), shape=(self.model_dim,)
    )
    self.inputs = jax.random.normal(
        key=jax.random.key(1), shape=(self.batch_size, self.model_dim)
    )

    def loss_fn(params, example):
      # This loss just computes a squared L2 distance.
      return 0.5 * jnp.sum(jnp.square(params - example))

    def forward(params, state, rng_per_example, inputs):
      del state, rng_per_example  # Unused
      batch_losses = jax.vmap(loss_fn, in_axes=(None, 0))(params, inputs)
      metrics = typing.Metrics(
          per_example={'loss': batch_losses},
      )
      average_loss = jnp.mean(batch_losses)
      return average_loss, ({}, metrics)

    self.forward = forward
    self.tol = {'rtol': 1e-6, 'atol': 1e-6}

    mesh_devices = np.array(jax.devices())
    self.mesh = jax.sharding.Mesh(mesh_devices, (SPMD_AXIS_NAME,))

  @parameterized.named_parameters(_PER_EXAMPLE_GRAD_METHODS)
  def test_clipped_gradients(
      self, per_example_grad_method: grad_clipping.PerExampleGradMethod
  ):
    value_and_grad_fn = jax.value_and_grad(self.forward, has_aux=True)
    clipping_fn = grad_clipping.global_clipping(
        clipping_norm=self.clipping_norm
    )
    grad_fn = per_example_grad_method(
        value_and_grad_fn,
        clipping_fn,
        state_acc_strategies=grad_clipping_utils.Reject(),
    )
    grad_fn_naive = grad_clipped_per_sample_naive(
        self.forward,
        clipping_norm=self.clipping_norm,
    )
    state = {}
    prng_key = jax.random.key(0)
    with self.mesh:
      (loss_actual, aux_actual), grad_actual = jax.jit(grad_fn)(  # pylint: disable=not-callable
          self.params, state, prng_key, self.inputs
      )
      (loss_expected, aux_expected), grad_expected = jax.jit(grad_fn_naive)(  # pylint: disable=not-callable
          self.params, state, prng_key, self.inputs
      )

    chex.assert_trees_all_close(loss_actual, loss_expected, **self.tol)
    chex.assert_trees_all_close(grad_actual, grad_expected, **self.tol)
    chex.assert_trees_all_close(aux_actual, aux_expected, **self.tol)


if __name__ == '__main__':
  absltest.main()
