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

"""Calibrating DP hyper-parameters using the RDP accountant."""

import math
import numbers
from typing import Sequence, Tuple, Union
import warnings

import numpy as np
import scipy.optimize as sp_opt

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting.rdp import rdp_privacy_accountant


def calibrate_steps(
    target_epsilon: float,
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_examples: int,
    target_delta: float = 1e-5,
    initial_estimate: int = 4,
    initial_min_steps: int = 1,
    tol: float = 0.1,
) -> int:
  """Computes the number of steps to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    noise_multipliers: Noise multiplier. Float or list of pairs
      (t: int, nm: float) if the noise multiplier changes across steps.
      't' indicates step where noise_multiplier is set to 'nm'.
    batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
      noise multiplier changes across steps. 't' indicates step where batch_size
      is set to 'bs'.
    num_examples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    initial_estimate: An initial estimate of the number of steps.
    initial_min_steps: Minimum number of steps.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    steps: Number of steps.
  """

  def get_epsilon(steps):
    return compute_epsilon(
        noise_multipliers,
        batch_sizes,
        steps,
        num_examples,
        target_delta,
    )

  if get_epsilon(initial_min_steps) > target_epsilon:
    raise ValueError('Epsilon at initial_min_steps is too large. '
                     'Try increasing `target_epsilon`.')

  max_steps = initial_estimate
  min_steps = initial_min_steps
  while get_epsilon(max_steps) < target_epsilon:
    min_steps, max_steps = max_steps, 2*max_steps

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  opt_result = sp_opt.minimize_scalar(
      error_epsilon,
      bounds=(min_steps, max_steps),
      method='bounded',
      options={'xatol': tol},
  )
  assert opt_result.success

  return math.ceil(opt_result.x)


def compute_epsilon(
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_steps: int,
    num_examples: int,
    target_delta: float = 1e-5,
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
    num_examples: Number of training examples
    target_delta: Desired delta for the returned epsilon

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

  orders = np.array((
      list(np.linspace(1.01, 8, num=50))
      + list(range(8, 64))
      + list(np.linspace(65, 512, num=10, dtype=int))
  ))

  accountant = rdp_privacy_accountant.RdpAccountant(
      orders, privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE)

  for t, nm, bs in t_nm_and_bs:
    q = bs / float(num_examples)
    event = dp_event.PoissonSampledDpEvent(
        q, dp_event.GaussianDpEvent(nm))
    # The RDP accounting is accumulated in-place.
    accountant.compose(event, t)

  eps, unused_opt_order = accountant.get_epsilon_and_optimal_order(
      target_delta=target_delta)

  return eps


def calibrate_noise_multiplier(
    target_epsilon: float,
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_steps: int,
    num_examples: int,
    target_delta: float = 1e-5,
    tol: float = 0.01,
) -> float:
  """Computes the noise multiplier to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
      noise multiplier changes across steps. 't' indicates step where batch_size
      is set to 'bs'.
    num_steps: Total number of iterations.
    num_examples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    noise_multiplier: Noise multiplier.
  """

  def get_epsilon(noise_multiplier):
    return compute_epsilon(
        noise_multiplier,
        batch_sizes,
        num_steps,
        num_examples,
        target_delta)

  max_noise = 1.0
  min_noise = 0.0
  while get_epsilon(max_noise) > target_epsilon:
    min_noise, max_noise = max_noise, 2*max_noise

  error_epsilon = lambda s: np.abs(get_epsilon(s) - target_epsilon)
  opt_result = sp_opt.minimize_scalar(
      error_epsilon,
      bounds=(min_noise, max_noise),
      method='bounded',
      options={'xatol': tol},
  )
  assert opt_result.success

  return opt_result.x


def calibrate_batch_size(
    target_epsilon: float,
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    num_steps: int,
    num_examples: int,
    target_delta: float = 1e-5,
    tol: float = 0.01,
) -> int:
  """Computes the batch size required to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    noise_multipliers: Noise multiplier. Float or list of pairs
      (t: int, nm: float) if the noise multiplier changes across steps.
      't' indicates step where noise_multiplier is set to 'nm'.
    num_steps: Total number of iterations.
    num_examples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    batch_size: Batch size.
  """

  def get_epsilon(batch_size):
    return compute_epsilon(
        noise_multipliers,
        batch_size,
        num_steps,
        num_examples,
        target_delta)

  max_batch_size = 8
  min_batch_size = 1

  if get_epsilon(min_batch_size) > target_epsilon:
    raise ValueError('Epsilon at batch size 1 is too large. '
                     'Try increasing `target_epsilon`.')

  while get_epsilon(max_batch_size) < target_epsilon:
    min_batch_size, max_batch_size = max_batch_size, 2*max_batch_size

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  opt_result = sp_opt.minimize_scalar(
      error_epsilon,
      bounds=(min_batch_size, max_batch_size),
      method='bounded',
      options={'xatol': tol},
  )
  assert opt_result.success

  return math.floor(opt_result.x)


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
