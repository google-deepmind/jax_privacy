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

"""Compute bounds on DP epsilon (given delta) for DP-SGD."""

import abc
import dataclasses
import numbers
from typing import Sequence, Tuple, Union
import warnings

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

  orders: Sequence[int] = dataclasses.field(
      default_factory=lambda: _DEFAULT_RDP_ORDERS)

  def __post_init__(self):
    self.orders = np.array(self.orders)

  def create_accountant(self) -> dp_accounting.rdp.RdpAccountant:
    return dp_accounting.rdp.RdpAccountant(
        orders=self.orders,
        neighboring_relation=(
            dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE),
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class PldAccountantConfig(DpAccountantConfig):

  # Values smaller than 1e-5 can result in slower and less accurate accounting.
  # b/251010738
  value_discretization_interval: float = _DEFAULT_PLD_DISCRETIZATION

  def create_accountant(self) -> dp_accounting.pld.PLDAccountant:
    return dp_accounting.pld.PLDAccountant(
        neighboring_relation=(
            dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE),
        value_discretization_interval=self.value_discretization_interval,
    )


def compute_epsilon(
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_steps: int,
    num_examples: int,
    target_delta: float,
    dp_accountant_config: DpAccountantConfig,
) -> float:
  """Computes epsilon for heterogeneous noise and mini-batch sizes.

  Args:
    noise_multipliers: Noise multiplier. Float or list of pairs
      (t: int, nm: float) if the noise multiplier changes across steps.
      't' indicates step where noise_multiplier is set to 'nm'.
    batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
      noise multiplier changes across steps. 't' indicates step where batch_size
      is set to 'bs'.
    num_steps: Total number of iterations.
    num_examples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    dp_accountant_config: Configuration for the DP accountant to use.

  Returns:
    epsilon: Privacy spent.
  """

  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')

  # If noise_multipliers is a number, turn it into list format of (0, nm).
  if isinstance(noise_multipliers, numbers.Number):
    noise_multipliers = [(0, noise_multipliers)]
  # If batch_sizes is a number, turn it into list format of (0, bs).
  if isinstance(batch_sizes, int):
    batch_sizes = [(0, batch_sizes)]

  # Make sure the time steps of changes are increasing.
  noise_multipliers = sorted(noise_multipliers, key=lambda t: t[0])
  batch_sizes = sorted(batch_sizes, key=lambda x: x[0])
  # Make sure the first time step is 0 in both sequences of hyper-parameters.
  assert noise_multipliers[0][0] == 0
  assert batch_sizes[0][0] == 0
  # Remove any settings which occur later than the maximum number of steps.
  noise_multipliers = [(t, x) for t, x in noise_multipliers if t <= num_steps]
  batch_sizes = [x for x in batch_sizes if x[0] <= num_steps]

  # Interleave both sequences of hyper-parameters into a single one.
  nm_and_bs = _interleave(noise_multipliers, batch_sizes)
  t_nm_and_bs = []
  # Adjust time indices to count number of steps in each configuration.
  for i in range(len(nm_and_bs) - 1):
    t_nm_and_bs.append((nm_and_bs[i + 1][0] - nm_and_bs[i][0], nm_and_bs[i][1],
                        nm_and_bs[i][2]))
  t_nm_and_bs.append(
      (num_steps - nm_and_bs[-1][0], nm_and_bs[-1][1], nm_and_bs[-1][2]))

  dp_accountant = dp_accountant_config.create_accountant()

  for t, nm, bs in t_nm_and_bs:
    q = bs / float(num_examples)
    event = dp_accounting.PoissonSampledDpEvent(
        q, dp_accounting.GaussianDpEvent(nm))
    dp_accountant.compose(event, t)

  eps = dp_accountant.get_epsilon(target_delta=target_delta)

  return eps


def _interleave(t_a, t_b):
  """Helper function to pair two timed sequences."""
  ts = [t for (t, _) in t_a] + [t for (t, _) in t_b]
  ts = list(set(ts))
  ts.sort()
  def _find_pair(t):
    a = [a for (s, a) in t_a if s <= t][-1]
    b = [b for (s, b) in t_b if s <= t][-1]
    return a, b
  return [(t, *_find_pair(t)) for t in ts]
