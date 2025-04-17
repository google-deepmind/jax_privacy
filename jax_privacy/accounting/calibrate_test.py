# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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

from absl.testing import absltest
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate
import numpy as np


_BATCH_SIZE = 1024
_NOISE_MULTIPLIER = 4.0
_NUM_SAMPLES = 50_000
_EPSILON = 2.27535
_EPSILON_USER = 5.00008
_DELTA = 1e-5
_NUM_UPDATES = 10_000
_EXAMPLES_PER_USER = 2


class CalibrateTest(absltest.TestCase):

  def test_calibrate_noise(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=_EPSILON,
        accountant=accountant,
        batch_sizes=_BATCH_SIZE,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        tol=1e-4,
    )

    np.testing.assert_allclose(noise_multiplier, _NOISE_MULTIPLIER, rtol=1e-4)

    dp_params = analysis.DpParams(
        noise_multipliers=noise_multiplier,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)

  def test_calibrate_batch_size(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    batch_size = calibrate.calibrate_batch_size(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
    )

    self.assertLessEqual(np.abs(batch_size - _BATCH_SIZE), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=batch_size,
        delta=_DELTA,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-2)

  def test_calibrate_num_updates(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    num_updates = calibrate.calibrate_num_updates(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON,
        batch_sizes=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
    )

    self.assertLessEqual(np.abs(num_updates - _NUM_UPDATES), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
    )
    epsilon = accountant.compute_epsilon(num_updates, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)

  def test_calibrate_noise_user_level(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=_EPSILON_USER,
        accountant=accountant,
        batch_sizes=_BATCH_SIZE,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
        tol=1e-4,
    )

    np.testing.assert_allclose(noise_multiplier, _NOISE_MULTIPLIER, rtol=1e-4)

    dp_params = analysis.DpParams(
        noise_multipliers=noise_multiplier,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_USER, rtol=1e-4)

  def test_calibrate_batch_size_user_level(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    batch_size = calibrate.calibrate_batch_size(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON_USER,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )

    self.assertLessEqual(np.abs(batch_size - _BATCH_SIZE), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=batch_size,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_USER, rtol=1e-2)

  def test_calibrate_num_updates_user_level(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    num_updates = calibrate.calibrate_num_updates(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON_USER,
        batch_sizes=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )

    self.assertLessEqual(np.abs(num_updates - _NUM_UPDATES), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    epsilon = accountant.compute_epsilon(num_updates, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_USER, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
