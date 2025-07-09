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

from collections.abc import Callable
import dataclasses
import os
import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import typing
from jax_privacy.training import algorithm_config
import numpy as np


class KbParticipationTest(parameterized.TestCase):

  def assert_fn_raises_valueerror_nonjit_and_jit(
      self,
      fn: Callable[[jnp.ndarray], Any],
      inp: jnp.ndarray,
  ):
    """Asserts that fn raises ValueError in non-jit and jit contexts."""
    with self.assertRaises(ValueError):
      fn(inp)

    # The callback may be reordered, so other errors may come first. However,
    # the callback should still provide a useful output even if the actual error
    # raised is someting else.
    jit_fn = jax.jit(fn)
    with self.assertRaises(Exception):
      jit_fn(inp)  # pylint: disable=not-callable

  @parameterized.named_parameters(
      ['scalar', 1], ['vector', 10], ['tensor', (10, 10, 10)]
  )
  def test_raises_nonmatrix(self, shape):
    tensor = jnp.ones(shape)
    sens_fn = algorithm_config.KbParticipation(k=1).sensitivity
    self.assert_fn_raises_valueerror_nonjit_and_jit(sens_fn, tensor)

  @parameterized.named_parameters(
      ['wide_rectangular', (5, 10)], ['long_rectangular', (10, 5)]
  )
  def test_raises_nonsquare(self, shape):
    tensor = jnp.ones(shape)
    sens_fn = algorithm_config.KbParticipation(k=1).sensitivity
    self.assert_fn_raises_valueerror_nonjit_and_jit(sens_fn, tensor)

  @parameterized.parameters([3], [4], [6], [7], [8], [9])
  def test_raises_nondivisor(self, k):
    tensor = jnp.ones((10, 10))
    sens_fn = algorithm_config.KbParticipation(k=k).sensitivity
    self.assert_fn_raises_valueerror_nonjit_and_jit(sens_fn, tensor)

  @parameterized.parameters([10], [20], [30])
  def test_sensitivity_correct_single_epoch(self, seed):
    rng = jax.random.PRNGKey(seed)
    num_steps = 10
    encoder = jax.random.uniform(rng, (num_steps, num_steps), dtype=jnp.float32)
    sensitivity_config = algorithm_config.KbParticipation(k=1)

    max_col_norm = jnp.max(jnp.linalg.norm(encoder, axis=0))
    np.testing.assert_allclose(
        sensitivity_config.sensitivity(encoder),
        max_col_norm,
        rtol=1e-6,
        atol=1e-6,
    )

  @parameterized.product(seed=[2, 5, 10], num_steps=[2, 5, 10])
  def test_sensitivity_correct_everystep(self, seed, num_steps):
    rng = jax.random.PRNGKey(seed)
    encoder = jax.random.uniform(rng, (num_steps, num_steps), dtype=jnp.float32)
    sensitivity_config = algorithm_config.KbParticipation(k=num_steps)

    actual = sensitivity_config.sensitivity(encoder)
    expected = jnp.linalg.norm(encoder.sum(axis=1))
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

  @parameterized.product(k=[1, 2, 4, 8], coefficient=[1, 2, 5, 10])
  def test_identity_encoder(self, k, coefficient):
    encoder = jnp.eye(8) * coefficient
    sensitivity_config = algorithm_config.KbParticipation(k=k)

    actual = sensitivity_config.sensitivity(encoder)
    expected = jnp.sqrt(k) * coefficient
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class _TestSensitivityConfig(algorithm_config.SensitivityConfig):
  fake_arg: int = 1

  def sensitivity(self, encoder_matrix: typing.SquareMatrix) -> float:
    raise NotImplementedError()


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class _TestMatrixMechanismConfig(algorithm_config.MatrixMechanismConfig):
  fake_arg: int = 1

  def correlation_matrix(self):
    raise NotImplementedError()


class MatrixMechanismConfigTest(parameterized.TestCase):

  def test_cachename_uses_cache(self):
    tmpdir = self.create_tempdir('cache_dir')
    full_cache_path = tmpdir.full_path
    sensitivity_config = _TestSensitivityConfig()
    config = _TestMatrixMechanismConfig(
        num_updates=1,
        sensitivity_config=sensitivity_config,
        cache_base_path=full_cache_path,
    )

    cache_check = re.compile(f'{full_cache_path}.*')
    self.assertTrue(cache_check.match(config.cache_path))

  @parameterized.named_parameters([
      ('matrix_config_changed', {'fake_arg': 2}, {}),
      ('sensitivity_config_changed', {}, {'fake_arg': 2}),
  ])
  def test_cachname_uses_args(
      self, matrix_mechanism_kwargs, sensitivity_config_kwargs
  ):
    tmpdir = self.create_tempdir('cache_dir')
    full_cache_path = tmpdir.full_path
    sensitivity_config = _TestSensitivityConfig()
    config = _TestMatrixMechanismConfig(
        num_updates=1,
        sensitivity_config=sensitivity_config,
        cache_base_path=full_cache_path,
    )
    initial_cache_name = config.cache_path

    sensitivity_config = _TestSensitivityConfig(**sensitivity_config_kwargs)
    config = _TestMatrixMechanismConfig(
        num_updates=1,
        sensitivity_config=sensitivity_config,
        cache_base_path=full_cache_path,
        **matrix_mechanism_kwargs,
    )
    self.assertNotEqual(config.cache_path, initial_cache_name)

  def test_manual_overwrites_cachename(self):
    manual_overwrite = 'test'
    tmpdir = self.create_tempdir('cache_dir')
    full_cache_path = tmpdir.full_path
    sensitivity_config = _TestSensitivityConfig()
    config = _TestMatrixMechanismConfig(
        num_updates=1,
        sensitivity_config=sensitivity_config,
        cache_base_path=full_cache_path,
        manual_cache_file_name=manual_overwrite,
    )

    filename = os.path.basename(config.cache_path).split('.')[0]
    self.assertEqual(filename, f'{manual_overwrite}')


if __name__ == '__main__':
  absltest.main()
