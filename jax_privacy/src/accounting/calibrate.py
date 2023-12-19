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

"""Calibrating DP hyper-parameters using the RDP accountant."""

import math
from typing import Sequence, Tuple, Union

from jax_privacy.src.accounting import dp_bounds
import numpy as np
import scipy.optimize as sp_opt


def calibrate_steps(
    target_epsilon: float,
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_examples: int,
    target_delta: float,
    dp_accountant_config: dp_bounds.DpAccountantConfig,
    initial_max_steps: int = 4,
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
    dp_accountant_config: Configuration for the DP accountant to use.
    initial_max_steps: An initial estimate of the number of steps.
    initial_min_steps: Minimum number of steps.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    steps: Number of steps.
  """

  def get_epsilon(num_steps):
    return dp_bounds.compute_epsilon(
        noise_multipliers=noise_multipliers,
        batch_sizes=batch_sizes,
        num_steps=num_steps,
        num_examples=num_examples,
        target_delta=target_delta,
        dp_accountant_config=dp_accountant_config,
    )

  if get_epsilon(initial_min_steps) > target_epsilon:
    raise ValueError('Epsilon at initial_min_steps is too large. '
                     'Try increasing `target_epsilon`.')

  max_steps = initial_max_steps
  min_steps = initial_min_steps
  while get_epsilon(max_steps) < target_epsilon:
    min_steps, max_steps = max_steps, 2*max_steps

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  steps = int(
      math.ceil(_solve_calibration(error_epsilon, min_steps, max_steps, tol)))

  return steps


def calibrate_noise_multiplier(
    target_epsilon: float,
    batch_sizes: Union[int, Sequence[Tuple[int, int]]],
    num_steps: int,
    num_examples: int,
    target_delta: float,
    dp_accountant_config: dp_bounds.DpAccountantConfig,
    initial_max_noise: float = 1.0,
    initial_min_noise: float = 0.0,
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
    dp_accountant_config: Configuration for the DP accountant to use.
    initial_max_noise: An initial estimate of the noise multiplier.
    initial_min_noise: Minimum noise multiplier.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    noise_multiplier: Noise multiplier.
  """

  def get_epsilon(noise_multiplier):
    return dp_bounds.compute_epsilon(
        noise_multipliers=noise_multiplier,
        batch_sizes=batch_sizes,
        num_steps=num_steps,
        num_examples=num_examples,
        target_delta=target_delta,
        dp_accountant_config=dp_accountant_config,
    )

  max_noise = initial_max_noise
  min_noise = initial_min_noise
  while get_epsilon(max_noise) > target_epsilon:
    min_noise, max_noise = max_noise, 2*max_noise

  error_epsilon = lambda s: np.abs(get_epsilon(s) - target_epsilon)
  noise_multiplier = float(
      _solve_calibration(error_epsilon, min_noise, max_noise, tol))

  return noise_multiplier


def calibrate_batch_size(
    target_epsilon: float,
    noise_multipliers: Union[float, Sequence[Tuple[int, float]]],
    num_steps: int,
    num_examples: int,
    target_delta: float,
    dp_accountant_config: dp_bounds.DpAccountantConfig,
    initial_max_batch_size: int = 8,
    initial_min_batch_size: int = 1,
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
    dp_accountant_config: Configuration for the DP accountant to use.
    initial_max_batch_size: An initial estimate of the batch size.
    initial_min_batch_size: Minimum batch size.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    batch_size: Batch size.
  """

  def get_epsilon(batch_size):
    return dp_bounds.compute_epsilon(
        noise_multipliers=noise_multipliers,
        batch_sizes=batch_size,
        num_steps=num_steps,
        num_examples=num_examples,
        target_delta=target_delta,
        dp_accountant_config=dp_accountant_config,
    )

  max_batch_size = initial_max_batch_size
  min_batch_size = initial_min_batch_size

  if get_epsilon(min_batch_size) > target_epsilon:
    raise ValueError('Epsilon at batch size 1 is too large. '
                     'Try increasing `target_epsilon`.')

  while get_epsilon(max_batch_size) < target_epsilon:
    min_batch_size, max_batch_size = max_batch_size, 2*max_batch_size

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  batch_size = int(math.floor(
      _solve_calibration(error_epsilon, min_batch_size, max_batch_size, tol)))

  return batch_size


def _solve_calibration(fn, x_min, x_max, tol):
  opt_result = sp_opt.minimize_scalar(
      fn,
      bounds=(x_min, x_max),
      method='bounded',
      options={'xatol': tol},
  )
  assert opt_result.success

  return opt_result.x
