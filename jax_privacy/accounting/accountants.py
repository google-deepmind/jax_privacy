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

"""Compute bounds on DP epsilon (given delta) for DP-SGD."""

import abc
from collections.abc import Sequence
import dataclasses

import dp_accounting
import numpy as np


_DEFAULT_RDP_ORDERS = np.concatenate((
    np.linspace(1.01, 8, num=50),
    np.arange(8, 64),
    np.linspace(65, 512, num=10, dtype=int),
))

_DEFAULT_PLD_DISCRETIZATION = 1e-4


class DpAccountantConfig(metaclass=abc.ABCMeta):
  """Configuration for the DP Accountant to use."""

  @abc.abstractmethod
  def create_accountant(self) -> dp_accounting.PrivacyAccountant:
    """Creates an accountant (with a new state)."""


@dataclasses.dataclass(kw_only=True, slots=True)
class RdpAccountantConfig(DpAccountantConfig):
  """Configuration for the RDP Accountant to use."""

  orders: Sequence[int] = dataclasses.field(
      default_factory=lambda: _DEFAULT_RDP_ORDERS
  )

  def __post_init__(self):
    self.orders = np.array(self.orders)

  def create_accountant(self) -> dp_accounting.rdp.RdpAccountant:
    return dp_accounting.rdp.RdpAccountant(
        orders=self.orders,
        neighboring_relation=(
            dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
        ),
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class PldAccountantConfig(DpAccountantConfig):
  """Configuration for the PLD Accountant to use."""

  # Values smaller than 1e-5 can result in slower and less accurate accounting.
  # b/251010738
  value_discretization_interval: float = _DEFAULT_PLD_DISCRETIZATION

  def create_accountant(self) -> dp_accounting.pld.PLDAccountant:
    return dp_accounting.pld.PLDAccountant(
        neighboring_relation=(
            dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
        ),
        value_discretization_interval=self.value_discretization_interval,
    )
