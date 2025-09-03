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

"""Device layout abstraction."""

from collections.abc import Mapping, Sequence
from typing import Any

import chex
import jax


class DeviceLayout:
  """Common args to `pmap` and `psum` for data parallelism."""

  def __init__(
      self,
      *,
      pmap_axis_name: str = 'data',
      devices: Sequence[jax.Device] | None = None,
  ):
    """Constructor.

    Args:
      pmap_axis_name: Parallel mapping axis name, to pass to `jax.pmap`.
      devices: XLA devices to pass to `jax.pmap`.
    """
    self.pmap_axis_name = pmap_axis_name
    self.devices = devices

  @property
  def pmap_kwargs(self) -> Mapping[str, Any]:
    return {
        'devices': self.devices,
        'axis_name': self.pmap_axis_name,
    }

  @property
  def data_psum_kwargs(self) -> Mapping[str, Any]:
    return {
        'axis_name': self.pmap_axis_name,
        'axis_index_groups': None,
    }

  @property
  def replica_index(self) -> chex.Array:
    """Index of the replica (to be called under a `pmap`)."""
    return jax.lax.axis_index(self.pmap_axis_name)
