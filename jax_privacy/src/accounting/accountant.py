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

"""Keeping track of the differential privacy guarantee."""

from typing import Optional

from jax_privacy.src.accounting import calibrate
from jax_privacy.src.training import batching as batching_py
import numpy as np


class Accountant:
  """Accountant for the differential privacy guarantee."""

  def __init__(
      self,
      clipping_norm: Optional[float],
      std_relative: Optional[float],
      dp_epsilon: float,
      dp_delta: float,
      num_samples: int,
      batching: batching_py.VirtualBatching,
  ):
    """Initializes the accountant for Differential Privacy.

    This class wraps the functions defined in `calibrate.py` so that we can
    gracefully handle limit cases where either `std_relative=0` or the clipping
    norm is infinite, which both result in infinite (vacuous) DP guarantees.

    Args:
      clipping_norm: clipping-norm for the per-example gradients (before
        averaging across the examples of the mini-batch).
      std_relative: standard deviation relative to the clipping-norm (aka noise
        multiplier).
      dp_epsilon: epsilon-value of DP guarantee.
      dp_delta: delta-value of DP guarantee.
      num_samples: number of examples in the training set.
      batching: batch-size used during training.
    """
    if clipping_norm is None:
      self._clipping_norm = float('inf')
    elif clipping_norm < 0:
      raise ValueError('Clipping norm must be non-negative.')
    else:
      self._clipping_norm = clipping_norm

    if std_relative is None:
      self._std_relative = 0
    elif std_relative < 0:
      raise ValueError('Standard deviation must be non-negative.')
    else:
      self._std_relative = std_relative

    self._batching = batching
    self._num_samples = num_samples

    self._dp_epsilon = dp_epsilon
    self._dp_delta = dp_delta

    self._batch_sizes = [(0, self._batching.batch_size_init)]
    if self._batching.scale_schedule is not None:
      self._batch_sizes.extend(
          [(threshold, self._batching.batch_size(threshold+1))
           for threshold in self._batching.scale_schedule]
      )

  def finite_dp_guarantee(self) -> bool:
    """Returns whether the DP guarantee (eps, delta) can be finite."""
    # The privacy (eps, delta) can only be finite with non-zero noise
    # and with a finite clipping-norm.
    return bool(self._std_relative and np.isfinite(self._clipping_norm))

  def compute_max_num_updates(self) -> int:
    """Compute maximum number of updates given the DP parameters."""
    if self.finite_dp_guarantee():
      return calibrate.calibrate_steps(
          target_epsilon=self._dp_epsilon,
          noise_multipliers=self._std_relative,
          batch_sizes=self._batch_sizes,
          num_examples=self._num_samples,
          target_delta=self._dp_delta,
      )
    else:
      return 0

  def compute_current_epsilon(self, num_updates: int) -> float:
    """Compute DP epsilon given the DP parameters and current `num_updates`."""
    if num_updates == 0:
      return 0.0
    elif self.finite_dp_guarantee():
      return calibrate.compute_epsilon(
          num_steps=num_updates,
          noise_multipliers=self._std_relative,
          batch_sizes=self._batch_sizes,
          num_examples=self._num_samples,
          target_delta=self._dp_delta,
      )
    else:
      return float('inf')
