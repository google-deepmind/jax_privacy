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
from absl.testing import parameterized
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
import numpy as np

_BATCH_SIZE = 1024
_NOISE_MULTIPLIER = 4.0
_NUM_SAMPLES = 50_000
_EXAMPLES_PER_USER = 2
_EPSILON_RDP = 2.27535
_EPSILON_PLD = 2.09245
_EPSILON_PLD_FBS = 4.89011
_EPSILON_PLD_USER = 5.00008
_EPSILON_PLD_USER_FBS = 11.48878
_DELTA = 1e-5
_NUM_UPDATES = 10_000
_CYCLE_LENGTH = 64


class DpsgdTrainingAccountantTest(parameterized.TestCase):

  def test_compute_epsilon_via_rdp(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  def test_compute_epsilon_via_pld(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD, rtol=1e-5)

  def test_compute_epsilon_fixed_batch_size(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER * 2,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  @parameterized.parameters(
      analysis.DpParams(
          noise_multipliers=None,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES,
          delta=_DELTA,
          examples_per_user=1,
      ),
      analysis.DpParams(
          noise_multipliers=_NOISE_MULTIPLIER,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES,
          delta=_DELTA,
          examples_per_user=2,
      ),
  )
  def test_validate_dp_params_raises_error(self, dp_params):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)


class DpsgdTrainingAccountantMaxBatchSizeTest(parameterized.TestCase):
  """Tests for DpsgdTrainingAccountant with truncated_batch_size != None."""

  @parameterized.parameters(1, 2, 4, 8, 16)
  def test_compute_epsilon_via_pld(self, truncated_batch_size_multiplier):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    truncated_batch_size = truncated_batch_size_multiplier * _BATCH_SIZE
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        truncated_batch_size=truncated_batch_size,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    # This analysis is not necessarily tight, so we check that epsilon is
    # between the tight values for Poisson sampling and fixed batch size
    # sampling.
    self.assertBetween(epsilon, _EPSILON_PLD, _EPSILON_PLD_FBS)

  def test_compute_epsilon_via_pld_high_truncated_batch_size(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    # Since we set truncated_batch_size to num_samples, truncation never happens
    # so the epsilon is the same as the case where truncated_batch_size is None.
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        truncated_batch_size=_NUM_SAMPLES,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD, rtol=1e-5)

  def test_compute_epsilon_via_pld_high_sampling_prob(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    # If we set batch_size = num_samples and truncated_batch_size = _BATCH_SIZE,
    # this is the same as fixed batch size sampling, which is the same as
    # Poisson sampling without truncation if we halve the noise.
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER * 2,
        batch_size=_NUM_SAMPLES,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        truncated_batch_size=_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD, rtol=1e-5)

  def test_validate_dp_params_raises_error(self):
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        examples_per_user=1,
        truncated_batch_size=_BATCH_SIZE,
    )
    accountant_config = accountants.RdpAccountantConfig()
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountant_config
    )
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)


class DpsgdTrainingAccountantCycleLengthTest(parameterized.TestCase):
  """Tests for DpsgdTrainingAccountant with cycle_length != 1.

  For most tests, we multiply num_samples and num_updates by _CYCLE_LENGTH. This
  gives the same expected output as DpsgdTrainingAccountantTest.
  """

  def test_compute_epsilon_via_rdp(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES * _CYCLE_LENGTH, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  def test_compute_epsilon_via_pld(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        delta=_DELTA,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES * _CYCLE_LENGTH, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD, rtol=1e-5)

  def test_compute_epsilon_fixed_batch_size(self):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER * 2,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
        delta=_DELTA,
        sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
        cycle_length=_CYCLE_LENGTH,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES * _CYCLE_LENGTH, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  @parameterized.parameters(
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER * 2,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES * _CYCLE_LENGTH + _CYCLE_LENGTH - 1,
              delta=_DELTA,
              sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
              cycle_length=_CYCLE_LENGTH,
          ),
          _NUM_UPDATES * _CYCLE_LENGTH,
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER * 2,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
              delta=_DELTA,
              sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
              cycle_length=_CYCLE_LENGTH,
          ),
          (_NUM_UPDATES - 1) * _CYCLE_LENGTH + 1,
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER * 2,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES * _CYCLE_LENGTH + _CYCLE_LENGTH - 1,
              delta=_DELTA,
              sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
              cycle_length=_CYCLE_LENGTH,
          ),
          (_NUM_UPDATES - 1) * _CYCLE_LENGTH + 1,
      ),
  )
  def test_uneven_multiples(self, dp_params, num_updates):
    """Test when num_samples/num_updates aren't multiples of cycle_length."""
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    epsilon = accountant.compute_epsilon(
        num_updates=num_updates, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_RDP, rtol=1e-5)

  @parameterized.parameters(
      analysis.DpParams(
          noise_multipliers=None,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
          delta=_DELTA,
          cycle_length=_CYCLE_LENGTH,
      ),
      # pytype: disable=wrong-arg-types
      analysis.DpParams(
          noise_multipliers=_NOISE_MULTIPLIER,
          # TODO: b/430253035 - Figure out whether list of tuples should be
          # supported.
          batch_size=[(10, _BATCH_SIZE), (10, _BATCH_SIZE * 2)],
          num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
          delta=_DELTA,
          cycle_length=_CYCLE_LENGTH,
      ),
      # pytype: enable=wrong-arg-types
      analysis.DpParams(
          noise_multipliers=[
              (10, _NOISE_MULTIPLIER),
              (10, _NOISE_MULTIPLIER * 2),
          ],
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
          delta=_DELTA,
          cycle_length=_CYCLE_LENGTH,
      ),
      analysis.DpParams(
          noise_multipliers=_NOISE_MULTIPLIER,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES * _CYCLE_LENGTH,
          delta=_DELTA,
          examples_per_user=2,
          cycle_length=_CYCLE_LENGTH,
      ),
      analysis.DpParams(
          noise_multipliers=_NOISE_MULTIPLIER,
          batch_size=_BATCH_SIZE,
          num_samples=_CYCLE_LENGTH * _BATCH_SIZE - 1,
          delta=_DELTA,
          cycle_length=_CYCLE_LENGTH,
      ),
  )
  def test_validate_dp_params_raises_error(self, dp_params):
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)


class SingleReleaseTrainingAccountantTest(parameterized.TestCase):

  @parameterized.parameters(
      [accountants.PldAccountantConfig()], [accountants.RdpAccountantConfig()]
  )
  def test_compute_epsilon_tigher_than_dpsgd(self, accountant_config):
    """Single release should always be tighter than amplification."""
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
    )

    single_release_accounter = analysis.SingleReleaseTrainingAccountant(
        dp_accountant_config=accountant_config
    )
    single_release_epsilon = single_release_accounter.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    dpsgd_accounter = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountant_config
    )
    dpsgd_epsilon = dpsgd_accounter.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    self.assertLess(single_release_epsilon, dpsgd_epsilon)

  @parameterized.parameters(
      analysis.DpParams(
          noise_multipliers=None,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES,
          delta=_DELTA,
          examples_per_user=1,
      ),
      analysis.DpParams(
          noise_multipliers=_NOISE_MULTIPLIER,
          batch_size=_BATCH_SIZE,
          num_samples=_NUM_SAMPLES,
          delta=_DELTA,
          examples_per_user=2,
      ),
  )
  def test_validate_dp_params_raises_error(self, dp_params):
    accountant = analysis.SingleReleaseTrainingAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)


class DpsgdTrainingUserLevelAccountantTest(parameterized.TestCase):

  def test_compute_epsilon_via_rdp_raises_error(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.RdpAccountantConfig()
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)

  def test_compute_epsilon_via_pld(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD_USER, rtol=1e-5)

  def test_compute_epsilon_fixed_batch_size(self):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(
        dp_accountant_config=accountants.PldAccountantConfig(
            value_discretization_interval=1e-2
        )
    )
    dp_params = analysis.DpParams(
        noise_multipliers=_NOISE_MULTIPLIER,
        batch_size=_BATCH_SIZE,
        num_samples=_NUM_SAMPLES,
        delta=_DELTA,
        examples_per_user=_EXAMPLES_PER_USER,
        sampling_method=analysis.SamplingMethod.FIXED_BATCH_SIZE,
    )
    epsilon = accountant.compute_epsilon(
        num_updates=_NUM_UPDATES, dp_params=dp_params
    )

    np.testing.assert_allclose(epsilon, _EPSILON_PLD_USER_FBS, rtol=1e-5)

  @parameterized.parameters(
      (
          analysis.DpParams(
              noise_multipliers=None,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES,
              delta=_DELTA,
              examples_per_user=1,
          ),
          accountants.PldAccountantConfig(value_discretization_interval=1e-2),
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES,
              delta=_DELTA,
              examples_per_user=None,
          ),
          accountants.PldAccountantConfig(value_discretization_interval=1e-2),
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES,
              delta=_DELTA,
              examples_per_user=1,
              cycle_length=2,
          ),
          accountants.PldAccountantConfig(value_discretization_interval=1e-2),
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES,
              delta=_DELTA,
              examples_per_user=1,
              truncated_batch_size=_BATCH_SIZE,
          ),
          accountants.PldAccountantConfig(value_discretization_interval=1e-2),
      ),
      (
          analysis.DpParams(
              noise_multipliers=_NOISE_MULTIPLIER,
              batch_size=_BATCH_SIZE,
              num_samples=_NUM_SAMPLES,
              delta=_DELTA,
              examples_per_user=1,
          ),
          accountants.RdpAccountantConfig(),
      ),
  )
  def test_validate_dp_params_raises_error(self, dp_params, accountant_config):
    accountant = analysis.DpsgdTrainingUserLevelAccountant(accountant_config)
    with self.assertRaises(ValueError):
      accountant.compute_epsilon(num_updates=_NUM_UPDATES, dp_params=dp_params)


if __name__ == '__main__':
  absltest.main()
