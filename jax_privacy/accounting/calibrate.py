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

"""Calibrating DP hyper-parameters using the RDP accountant."""

from collections.abc import Callable, Sequence
import math

from jax_privacy.accounting import analysis
import numpy as np
import scipy.optimize


def _solve_calibration(
    fn: Callable[[float], float], x_min: float, x_max: float, tol: float
) -> float:
  """Find an x in [x_min, x_max] that minimizes fn(x) using scipy.optimize."""
  opt_result = scipy.optimize.minimize_scalar(
      fn,
      bounds=(x_min, x_max),
      method='bounded',
      options={'xatol': tol},
  )
  assert opt_result.success

  return float(opt_result.x)


def calibrate_num_updates(
    *,
    target_epsilon: float,
    accountant: analysis.DpTrainingAccountant,
    noise_multipliers: float | Sequence[tuple[int, float]],
    batch_sizes: int | Sequence[tuple[int, int]],
    num_samples: int,
    target_delta: float,
    examples_per_user: int | None = None,
    cycle_length: int | None = None,
    truncated_batch_size: int | None = None,
    initial_max_updates: int = 4,
    initial_min_updates: int = 1,
    tol: float = 0.1,
) -> int:
  """Computes the number of steps to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    accountant: Method of computing the privacy guarantee.
    noise_multipliers: Noise multiplier. Float or list of pairs (t: int, nm:
      float) if the noise multiplier changes across updates. 't' indicates
      update where noise_multiplier is set to 'nm'.
    batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
      noise multiplier changes across updates. 't' indicates step where
      batch_size is set to 'bs'.
    num_samples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    examples_per_user: If multiple examples per user are used, this is the
      maximum number any user contributes to the training set.
    cycle_length: If using cyclic Poisson sampling with BandMF, the length of
      the cycle.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to.
    initial_max_updates: An initial estimate of the number of updates.
    initial_min_updates: Minimum number of updates.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    Number of updates.
  """
  if not accountant.can_calibrate_steps():
    raise ValueError(f'`accountant`={type(accountant)} cannot calibrate steps.')

  def get_epsilon(num_updates: int) -> float:
    dp_params = analysis.DpParams(
        noise_multipliers=noise_multipliers,
        batch_size=batch_sizes,
        num_samples=num_samples,
        delta=target_delta,
        examples_per_user=examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
    return accountant.compute_epsilon(num_updates, dp_params)

  if get_epsilon(initial_min_updates) > target_epsilon:
    raise ValueError(
        'Epsilon at initial_min_steps is too large. '
        'Try increasing `target_epsilon`.'
    )

  max_steps = initial_max_updates
  min_steps = initial_min_updates
  while get_epsilon(max_steps) < target_epsilon:
    min_steps, max_steps = max_steps, 2 * max_steps

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  steps = int(
      math.floor(_solve_calibration(error_epsilon, min_steps, max_steps, tol))
  )
  if cycle_length is not None and cycle_length != 1:
    # For BandMF, rounding up to the nearest multiple of cycle length does not
    # affect the privacy analysis. We should report this rounded up value to
    # the user so they can get more training steps for the same epsilon.
    return math.ceil(steps / cycle_length) * cycle_length
  else:
    return steps


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    accountant: analysis.DpTrainingAccountant,
    batch_sizes: int | Sequence[tuple[int, int]],
    num_updates: int,
    num_samples: int,
    target_delta: float,
    examples_per_user: int | None = None,
    cycle_length: int | None = None,
    truncated_batch_size: int | None = None,
    initial_max_noise: float = 1.0,
    initial_min_noise: float = 0.0,
    tol: float = 0.01,
) -> float:
  """Computes the noise multiplier to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    accountant: Method of computing the privacy guarantee.
    batch_sizes: Batch size. Integer or list of pairs (t: int, bs: int) if the
      noise multiplier changes across steps. 't' indicates step where batch_size
      is set to 'bs'.
    num_updates: Total number of iterations.
    num_samples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    examples_per_user: If multiple examples per user are used, this is the
      maximum number any user contributes to the training set.
    cycle_length: If using cyclic Poisson sampling with BandMF, the length of
      the cycle.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to.
    initial_max_noise: An initial estimate of the noise multiplier.
    initial_min_noise: Minimum noise multiplier.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    Noise multiplier.
  """
  if not accountant.can_calibrate_noise_multipliers():
    raise ValueError(
        f'`accountant`={type(accountant)} cannot calibrate noise multipliers.'
    )

  def get_epsilon(noise_multiplier: float) -> float:
    dp_params = analysis.DpParams(
        noise_multipliers=noise_multiplier,
        batch_size=batch_sizes,
        num_samples=num_samples,
        delta=target_delta,
        examples_per_user=examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
    return accountant.compute_epsilon(num_updates, dp_params)

  max_noise = initial_max_noise
  min_noise = initial_min_noise
  while get_epsilon(max_noise) > target_epsilon:
    min_noise, max_noise = max_noise, 2 * max_noise

  error_epsilon = lambda s: np.abs(get_epsilon(s) - target_epsilon)
  noise_multiplier = float(
      _solve_calibration(error_epsilon, min_noise, max_noise, tol)
  )

  return noise_multiplier


def calibrate_batch_size(
    *,
    target_epsilon: float,
    accountant: analysis.DpTrainingAccountant,
    noise_multipliers: float | Sequence[tuple[int, float]],
    num_updates: int,
    num_samples: int,
    target_delta: float,
    examples_per_user: int | None = None,
    cycle_length: int | None = None,
    truncated_batch_size: int | None = None,
    initial_max_batch_size: int = 8,
    initial_min_batch_size: int = 1,
    tol: float = 0.01,
) -> int:
  """Computes the batch size required to achieve `target_epsilon`.

  Args:
    target_epsilon: The desired final epsilon.
    accountant: Method of computing the privacy guarantee.
    noise_multipliers: Noise multiplier. Float or list of pairs (t: int, nm:
      float) if the noise multiplier changes across steps. 't' indicates step
      where noise_multiplier is set to 'nm'.
    num_updates: Total number of iterations.
    num_samples: Number of training examples.
    target_delta: Desired delta for the returned epsilon.
    examples_per_user: If multiple examples per user are used, this is the
      maximum number any user contributes to the training set.
    cycle_length: If using cyclic Poisson sampling with BandMF, the length of
      the cycle.
    truncated_batch_size: If using truncated Poisson sampling, the maximum batch
      size to truncate to.
    initial_max_batch_size: An initial estimate of the batch size.
    initial_min_batch_size: Minimum batch size.
    tol: tolerance of the optimizer for the calibration.

  Returns:
    Batch size.
  """
  if not accountant.can_calibrate_batch_size():
    raise ValueError(
        f'`accountant`={type(accountant)} cannot calibrate batch size.'
    )

  def get_epsilon(batch_size: int) -> float:
    dp_params = analysis.DpParams(
        noise_multipliers=noise_multipliers,
        batch_size=batch_size,
        num_samples=num_samples,
        delta=target_delta,
        examples_per_user=examples_per_user,
        cycle_length=cycle_length,
        truncated_batch_size=truncated_batch_size,
    )
    return accountant.compute_epsilon(num_updates, dp_params)

  max_batch_size = initial_max_batch_size
  min_batch_size = initial_min_batch_size

  if get_epsilon(min_batch_size) > target_epsilon:
    raise ValueError(
        'Epsilon at batch size 1 is too large. Try increasing `target_epsilon`.'
    )

  while get_epsilon(max_batch_size) < target_epsilon:
    min_batch_size, max_batch_size = max_batch_size, 2 * max_batch_size

  error_epsilon = lambda s: np.abs(get_epsilon(int(s)) - target_epsilon)
  batch_size = int(
      math.floor(
          _solve_calibration(error_epsilon, min_batch_size, max_batch_size, tol)
      )
  )

  return batch_size
