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

"""Tests for calibration of DP hyper-parameters using privacy accountant."""

from absl.testing import absltest
from jax_privacy.src.accounting import calibrate
from jax_privacy.src.accounting import dp_bounds
import numpy as np


_BATCH_SIZE = 1024
_NOISE_MULTIPLIER = 4.0
_NUM_EXAMPLES = 50_000
_EPSILON = 2.27535
_DELTA = 1e-5
_NUM_STEPS = 10_000


class CalibrateTest(absltest.TestCase):

  def test_calibrate_noise(self):
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=_EPSILON,
        batch_sizes=_BATCH_SIZE,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
        tol=1e-4,
    )

    np.testing.assert_allclose(noise_multiplier, _NOISE_MULTIPLIER, rtol=1e-4)

    epsilon = dp_bounds.compute_epsilon(
        noise_multipliers=noise_multiplier,
        batch_sizes=_BATCH_SIZE,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)

  def test_calibrate_batch_size(self):
    batch_size = calibrate.calibrate_batch_size(
        noise_multipliers=_NOISE_MULTIPLIER,
        target_epsilon=_EPSILON,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    self.assertLessEqual(np.abs(batch_size - _BATCH_SIZE), 1)

    epsilon = dp_bounds.compute_epsilon(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_sizes=batch_size,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-2)

  def test_calibrate_num_steps(self):
    num_steps = calibrate.calibrate_steps(
        noise_multipliers=_NOISE_MULTIPLIER,
        target_epsilon=_EPSILON,
        batch_sizes=_BATCH_SIZE,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    self.assertLessEqual(np.abs(num_steps - _NUM_STEPS), 1)

    epsilon = dp_bounds.compute_epsilon(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_sizes=_BATCH_SIZE,
        num_steps=num_steps,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
