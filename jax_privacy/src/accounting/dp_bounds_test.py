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

"""Tests for computation of DP bounds."""

from absl.testing import absltest
from jax_privacy.src.accounting import dp_bounds
import numpy as np


_BATCH_SIZE = 1024
_NOISE_MULTIPLIER = 4.0
_NUM_EXAMPLES = 50_000
_EPSILON_RDP = 2.27535
_EPSILON_PLD = 2.09245
_DELTA = 1e-5
_NUM_STEPS = 10_000


class DPBoundsTest(absltest.TestCase):

  def test_compute_epsilon_via_rdp(self):
    epsilon = dp_bounds.compute_epsilon(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_sizes=_BATCH_SIZE,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.RdpAccountantConfig(),
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  def test_compute_epsilon_via_pld(self):
    epsilon = dp_bounds.compute_epsilon(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_sizes=_BATCH_SIZE,
        num_steps=_NUM_STEPS,
        num_examples=_NUM_EXAMPLES,
        target_delta=_DELTA,
        dp_accountant_config=dp_bounds.PldAccountantConfig(),
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
