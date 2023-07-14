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

"""Support for testing pmapped operations."""

import contextlib
import functools
from unittest import mock

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.dp_sgd import devices


class PmapTestCase(chex.TestCase):
  """Test case that simulates a multi-device pmapped environment."""

  def setUp(self):
    super().setUp()

    self._device_layout = devices.DeviceLayout(pmap_axis_name='test')

  @contextlib.contextmanager
  def patch_collectives(self, axis_index=3):
    mock_axis_index = functools.partial(self._axis_index, axis_index=axis_index)
    with mock.patch('jax.lax.axis_index', new=mock_axis_index), \
        mock.patch('jax.lax.pmean', new=self._pmean), \
        mock.patch('jax.lax.psum', new=self._psum), \
        mock.patch('jax.lax.all_gather', new=self._all_gather):
      yield

  def _axis_index(self, axis_name, *, axis_index):
    self.assertEqual(axis_name, 'test')
    return axis_index

  def _psum(self, x, axis_name, *, axis_index_groups):
    """Patch to psum."""
    self.assertEqual(axis_name, 'test')
    self.assertIsNone(axis_index_groups)
    # Assume four devices, two of which have zeros.
    return jax.tree_map(lambda t: 2. * t, x)

  def _pmean(self, x, axis_name, *, axis_index_groups):
    """Patch to pmean."""
    self.assertEqual(axis_name, 'test')
    self.assertIsNone(axis_index_groups)
    # Assume four devices, two of which have zeros.
    return jax.tree_map(lambda t: t / 2., x)

  def _all_gather(self, x, axis_name, *, axis_index_groups, tiled=False):
    """Patch to all_gather."""
    self.assertEqual(axis_name, 'test')
    self.assertIsNone(axis_index_groups)
    # Assume four devices, two of which have zeros.
    mask = jnp.array([1., 0., 0., 1.])
    result = jax.tree_map(
        lambda t: t * jnp.expand_dims(mask, axis=list(range(1, 1 + t.ndim))), x)
    if tiled:
      # Merge parallelization and batching dimensions if tiled.
      result = jax.tree_map(
          lambda t: jnp.reshape(t, [-1, *t.shape[2:]]), result)
    return result
