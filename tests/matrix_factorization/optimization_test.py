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
import hypothesis
from hypothesis import strategies as st
import jax.numpy as jnp
from jax_privacy.matrix_factorization import optimization
from jax_privacy.matrix_factorization import test_utils
import numpy as np


test_utils.configure_hypothesis()


class OptimizationTest(absltest.TestCase):

  @hypothesis.given(xstar=st.floats(-1e6, 1e6), ystar=st.floats(-1e6, 1e6))
  def test_basic(self, xstar, ystar):
    def loss(params):
      x, y = params
      return (x - xstar) ** 2 + 10 * (y - ystar) ** 2

    x0 = jnp.array(0.0)
    y0 = jnp.array(0.0)
    x1, y1 = optimization.optimize(loss, params=(x0, y0))
    np.testing.assert_allclose(x1, xstar, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(y1, ystar, rtol=1e-6, atol=1e-8)

  def test_steps_and_callback(self):
    def loss(x):
      return jnp.abs(x - 3)

    max_steps = 7
    steps = 0

    def callback(info):
      nonlocal steps
      self.assertIsInstance(info, optimization.CallbackArgs)
      self.assertEqual(info.step, steps)
      steps += 1

    _ = optimization.optimize(
        loss,
        params=jnp.asarray(0.0),
        max_optimizer_steps=max_steps,
        callback=callback,
    )
    self.assertEqual(steps, max_steps)

  def test_value_and_grad(self):
    def loss_and_grad(x):
      return x**2, 2 * x

    x1 = optimization.optimize(
        loss_and_grad,
        params=jnp.asarray(10.0),
        grad=True,
    )
    np.testing.assert_allclose(x1, 0.0)


if __name__ == '__main__':
  absltest.main()
