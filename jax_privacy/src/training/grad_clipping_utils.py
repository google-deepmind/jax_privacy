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

"""Utils for grad_clipping.py."""

import jax
import jax.numpy as jnp


def _should_average_array(array_batched, array_vec):
  """Determine whether array_vec should be averaged based on array_batched.

  In `ShapeEvaluator`, both `batched_shapes` and `_grad_fn_vectorized` return
  trees of the form `((loss, aux), (param_grads, clipping_aux))`.
  This function will be invoked on all leaf arrays of their outputs.

  Expected shapes are as follows:

  | output                     | batched shape | vectorised shape | action  |
  | -------------------------- | ------------- | ---------------- | ------- |
  | loss                       | ()            | (b,)             | average |
  | aux - per-example elements | (b, ...)      | (b, 1, ...)      | stack   |
  | aux - aggregate elements   | (...)         | (b, ...)         | average |
  | param_grads                | (...)         | (b, ...)         | average |
  | clipping_aux               | (b, ...)      | (b, ...)         | stack   |

  This assumes that `aux` never contains any arrays whose leading dimension has
  length 1, unless the batch size is 1.

  Args:
    array_batched: Placeholder array arising from applying the loss value /
      clipped gradient computation to a batch of examples.
    array_vec: Placeholder array arising from applying the loss value /
      clipped gradient computation to a singleton example (batch size one),
      then vectorising.

  Returns:
    Whether the vectorised array should be averaged (as opposed to stacking
    per-example values), in order to match the batched array.
  """
  if array_vec.shape in (
      array_batched.shape,
      (*array_batched.shape[:1], 1, *array_batched.shape[1:]),
  ):
    # The batched shape is compatible with the vectorized shape, so there
    # is no need to average: the array was supposed to be vectorized in
    # the batched forward pass.
    return False
  elif array_vec.shape[1:] == array_batched.shape:
    # The vectorised array has an additional leading batch dimension,
    # so it should be averaged.
    return True
  else:
    raise ValueError(
        f'Invalid shapes {array_batched.shape} and {array_vec.shape}.')


class ShapeEvaluator:
  """Evaluate shapes to determine which outputs are per-example.

  Loss value / clipped gradient computation returns trees of the form
  `((loss, aux), (param_grads, clipping_aux))`, where `loss` is a scalar and
  the other elements are themselves trees.

  Shapes of this tree are evaluated in two ways: Once on a batch, and once
  on a single input (batch size one) vectorised over the batch. Comparing the
  shapes between these two methods allows us to infer which output arrays are
  per-example, and which are aggregated (so should be averaged).
  """

  def __init__(
      self,
      forward_fn,
      clipping_fn,
      grad_fn_vectorized,
  ):
    self._grad_fn = jax.value_and_grad(forward_fn, has_aux=True)
    self._clipping_fn = clipping_fn
    self._grad_fn_vectorized = grad_fn_vectorized

  def batched_shapes(self, params, inputs, network_state, rng):
    """Evaluate the expected shapes."""
    batch_size = jax.tree_leaves(inputs)[0].shape[0]
    out, grads = jax.eval_shape(
        self._grad_fn, params, inputs, network_state, rng)
    grads, aux = jax.eval_shape(self._clipping_fn, grads)
    # The auxiliary output of `clipped_fn` is always vectorized so that we
    # can log statistics per sample.
    vectorize = lambda x: jax.lax.broadcast(x, (batch_size,))
    aux = jax.eval_shape(lambda tree: jax.tree_map(vectorize, tree), aux)
    return out, (grads, aux)

  def vectorized_shapes(self, params, inputs, network_state, rng):
    """Evaluate the expected shapes when vectorizing the gradient."""
    return jax.eval_shape(
        self._grad_fn_vectorized, params, inputs, network_state, rng)

  def should_average(self, params, inputs, network_state, rng):
    """Detect whether arrays should be averaged or stacked based on shapes."""
    return jax.tree_map(
        _should_average_array,
        self.batched_shapes(params, inputs, network_state, rng),
        self.vectorized_shapes(params, inputs, network_state, rng),
    )


class LoopAccumulator:
  """Accumulate or stack values and grads over a loop."""

  def __init__(self, shape_evaluator):
    self._shape_evaluator = shape_evaluator
    self._should_average = None

  def initialize(self, *args):
    self._should_average = self._shape_evaluator.should_average(*args)
    return jax.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        self._shape_evaluator.batched_shapes(*args),
    )

  def accumulate(self, value_and_grad, value_and_grad_i, i, batch_size):
    """Running average or stack of `value_and_grad_i` into `value_and_grad`."""
    def leaf_fn(leaf_acc, leaf_new, should_average):
      if should_average:
        leaf_new = jnp.reshape(leaf_new, leaf_acc.shape)
        return leaf_acc + leaf_new / batch_size
      else:
        leaf_new = jnp.reshape(leaf_new, leaf_acc[i].shape)
        return leaf_acc.at[i].set(leaf_new)

    return jax.tree_map(
        leaf_fn, value_and_grad, value_and_grad_i, self._should_average)


class VmapReducer:
  """Reduce values and grads after a vmap."""

  def __init__(self, shape_evaluator):
    self._shape_evaluator = shape_evaluator

  def reduce(self, value_and_grads, *args):
    """Reduce and reshape `value_and_grads` based on expected shapes."""

    def maybe_average_and_reshape(array, should_average, expected_array):
      """Average if needed, and reshape to expected shape."""
      if should_average:
        array = jnp.mean(array, axis=0)
      return jnp.reshape(array, expected_array.shape)

    expected_shapes = self._shape_evaluator.batched_shapes(*args)
    should_average_tree = self._shape_evaluator.should_average(*args)
    return jax.tree_map(
        maybe_average_and_reshape,
        value_and_grads,
        should_average_tree,
        expected_shapes,
    )
