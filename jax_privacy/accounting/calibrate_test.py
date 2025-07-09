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

from absl.testing import absltest
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate
import numpy as np


_BATCH_SIZE = 1024
_TRUNCATED_BATCH_SIZE = 1056
_NOISE_MULTIPLIER = 4.0
_NUM_SAMPLES = 50_000
_EPSILON = 2.27535
_EPSILON_TRUNCATED = 3.09152  # Potentially not tight; corresponding tests
# should be updated if the analysis is tightened.
_EPSILON_USER = 5.00008
_DELTA = 1e-5
_NUM_UPDATES = 10_000
_EXAMPLES_PER_USER = 2
_CYCLE_LENGTH = 2


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

    # TODO: Make this a one-sided test.
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

    self.assertBetween(num_updates, _NUM_UPDATES - 1, _NUM_UPDATES)

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

    # TODO: Make this a one-sided test.
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

    self.assertBetween(num_updates, _NUM_UPDATES - 1, _NUM_UPDATES)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    epsilon = accountant.compute_epsilon(num_updates, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_USER, rtol=1e-4)

  def test_calibrate_noise_bandmf(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    # Multiplying num_updates and num_samples by cycle_length retrieves the same
    # epsilon as DP-SGD.
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=_EPSILON,
        accountant=accountant,
        batch_sizes=_BATCH_SIZE,
        num_updates=_NUM_UPDATES * _CYCLE_LENGTH,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        target_delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
        tol=1e-4,
    )

    np.testing.assert_allclose(noise_multiplier, _NOISE_MULTIPLIER, rtol=1e-4)

    dp_params = analysis.DpParams(
        noise_multipliers=noise_multiplier,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(
        _NUM_UPDATES * _CYCLE_LENGTH, dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)

  def test_calibrate_batch_size_bandmf(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    # Multiplying num_updates and num_samples by cycle_length retrieves the same
    # epsilon as DP-SGD.
    batch_size = calibrate.calibrate_batch_size(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON,
        num_updates=_NUM_UPDATES * _CYCLE_LENGTH,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        target_delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )

    # TODO: Make this a one-sided test.
    self.assertLessEqual(np.abs(batch_size - _BATCH_SIZE), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        batch_size=batch_size,
        delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(
        _NUM_UPDATES * _CYCLE_LENGTH, dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-2)

  def test_calibrate_num_updates_bandmf(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    # Multiplying num_updates and num_samples by cycle_length retrieves the same
    # epsilon as DP-SGD.
    num_updates = calibrate.calibrate_num_updates(
        noise_multipliers=_NOISE_MULTIPLIER,
        accountant=accountant,
        target_epsilon=_EPSILON,
        batch_sizes=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        target_delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )

    # Being off by 1 for DP-SGD is equivalent to being off by cycle_length for
    # BandMF.
    self.assertBetween(
        num_updates,
        (_NUM_UPDATES - 1) * _CYCLE_LENGTH,
        _NUM_UPDATES * _CYCLE_LENGTH,
    )

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(num_updates, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON, rtol=1e-4)

  def test_calibrate_noise_truncated(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=_EPSILON_TRUNCATED,
        accountant=accountant,
        batch_sizes=_BATCH_SIZE,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
        tol=1e-4,
    )

    np.testing.assert_allclose(noise_multiplier, _NOISE_MULTIPLIER, rtol=1e-4)

    dp_params = analysis.DpParams(
        noise_multipliers=noise_multiplier,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_TRUNCATED, rtol=1e-4)

  def test_calibrate_batch_size_truncated(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    batch_size = calibrate.calibrate_batch_size(
        target_epsilon=_EPSILON_TRUNCATED,
        accountant=accountant,
        noise_multipliers=_NOISE_MULTIPLIER,
        num_updates=_NUM_UPDATES,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
        tol=1e-4,
    )

    # TODO: Make this a one-sided test.
    self.assertLessEqual(np.abs(batch_size - _BATCH_SIZE), 1)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=batch_size,
        delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(_NUM_UPDATES, dp_params)

    # With truncation epsilon is more sensitive to the batch size, so we
    # allow a larger error tolerance because of off-by-1 errors in the
    # calibration.
    np.testing.assert_allclose(epsilon, _EPSILON_TRUNCATED, rtol=5e-2)

  def test_calibrate_num_updates_truncated(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    num_updates = calibrate.calibrate_num_updates(
        target_epsilon=_EPSILON_TRUNCATED,
        accountant=accountant,
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_sizes=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        target_delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
        tol=1e-4,
    )

    self.assertBetween(num_updates, _NUM_UPDATES - 1, _NUM_UPDATES)

    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        num_samples=_NUM_SAMPLES,
        batch_size=_BATCH_SIZE,
        delta=_DELTA,
        truncated_batch_size=_TRUNCATED_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(num_updates, dp_params)

    np.testing.assert_allclose(epsilon, _EPSILON_TRUNCATED, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
