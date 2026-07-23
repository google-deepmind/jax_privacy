# Copyright 2026 DeepMind Technologies Limited.
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

import math

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy.experimental.monte_carlo import delta_calculation
import numpy as np


class DeltaCalculationTest(parameterized.TestCase):

  @parameterized.parameters(
      (100, 1, 1 / 2, 1.0),
      (1000, 1, 1 / 2, 1.0),
      (3, 2, 1 / 3, 1 / 2),
      (6, 2, 1 / 3, 1 / 4),
      (2, 3, 1 / 4, 1 / 3),
      (4, 3, 1 / 4, 1 / 9),
  )
  def test_hoeffding_bound(self, num_samples, tau, delta, expected_bound):
    # Check that Hoeffding bound is correct for hand-calculable cases.
    self.assertAlmostEqual(
        delta_calculation._hoeffding_bound(num_samples, tau, delta),
        expected_bound,
        places=5,
    )

  @parameterized.parameters([10**-i for i in range(1, 17)])
  def test_overall_delta(self, base_delta):
    # Check that overall delta is decreasing in num_samples, and always in
    # (base_delta, 1].
    overall_delta_10000 = delta_calculation.get_overall_delta(10000, base_delta)
    overall_delta_10001 = delta_calculation.get_overall_delta(10001, base_delta)
    self.assertLessEqual(overall_delta_10000, 1.0)
    self.assertLess(overall_delta_10001, overall_delta_10000)
    self.assertLess(base_delta, overall_delta_10001)

  @parameterized.parameters([(10 ** (i + 2), 10**-i) for i in range(1, 17)])
  def test_base_delta(self, num_samples, target_delta):
    base_delta = delta_calculation.get_base_delta(num_samples, target_delta)
    overall_delta = delta_calculation.get_overall_delta(num_samples, base_delta)
    self.assertLessEqual(overall_delta, target_delta)
    self.assertAlmostEqual(overall_delta, target_delta, places=5)

  @parameterized.parameters([(10**i, 10**-i) for i in range(1, 17)])
  def test_num_samples_too_small(self, num_samples, target_delta):
    with self.assertRaises(ValueError):
      delta_calculation.get_base_delta(num_samples, target_delta)

  @parameterized.product(
      base_delta_multiplier=[0.5, 0.8, 0.9],
      target_delta=[10**-i for i in range(1, 15)],
  )
  def test_minimum_samples_to_calibrate(
      self, base_delta_multiplier, target_delta
  ):
    base_delta = base_delta_multiplier * target_delta
    num_samples = delta_calculation.minimum_samples_to_calibrate(
        base_delta, target_delta
    )
    self.assertLessEqual(
        delta_calculation.get_overall_delta(num_samples, base_delta),
        target_delta,
    )
    self.assertGreaterEqual(
        delta_calculation.get_base_delta(num_samples, target_delta),
        base_delta / 1.001,
    )
    # Make sure that using one less sample fails.
    try:
      delta_calculation.get_base_delta(num_samples - 1, target_delta)
      # If base_delta was achievable, then 1 fewer sample should not be enough
      # to achieve the target delta.
      self.assertGreater(
          delta_calculation.get_overall_delta(num_samples - 1, base_delta),
          target_delta,
      )
    except ValueError:
      # One fewer sample failed, as expected.
      pass

  @parameterized.named_parameters(
      ('all_at_most_epsilon', 3, [1, 2, 3], None, 0.0),
      ('all_greater_than_epsilon', 3, [3 + math.log(2), 1e9], None, 3 / 4),
      ('some_greater_than_epsilon', 3, [2, 3, 3 + math.log(2)], None, 1 / 6),
      (
          'all_greater_than_epsilon_with_counts',
          3,
          [3 + math.log(2), 1e9],
          [2, 1],
          2 / 3,
      ),
      (
          'some_greater_than_epsilon_with_counts',
          3,
          [2, 3, 3 + math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
      ('large_epsilon', 1000, [999, 1000, 1000 + math.log(2)], None, 1 / 6),
      ('epsilon_zero', 0, [-1, 0, math.log(2)], None, 1 / 6),
      (
          'large_epsilon_with_counts',
          1000,
          [999, 1000, 1000 + math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
      (
          'epsilon_zero_with_counts',
          0,
          [-1, 0, math.log(2)],
          [1, 1, 2],
          1 / 4,
      ),
  )
  def test_delta_from_epsilon_and_samples(
      self, epsilon, samples, counts, expected_delta
  ):
    delta = delta_calculation.delta_from_epsilon_and_samples(
        epsilon, samples, counts
    )
    self.assertAlmostEqual(delta, expected_delta, places=5)

  @parameterized.parameters(
      ([1, 2, 3], None),
      ([4, 5], None),
      ([2, 3, 4], None),
      ([2, 3, 4], [2, 1, 1]),
      ([2, 3, 4], [1, 1, 2]),
  )
  def test_composition_with_no_op_event(self, samples, counts):
    """No-op DP event should have same result as no event."""
    delta_1 = delta_calculation.delta_from_epsilon_and_samples(
        3, samples, counts, dp_accounting.NoOpDpEvent()
    )
    delta_2 = delta_calculation.delta_from_epsilon_and_samples(
        3, samples, counts
    )
    self.assertBetween(delta_1, delta_2 * (1 - 1e-7), delta_2 * (1 + 1e-7))

  @parameterized.parameters(
      ([1, 2, 3], None),
      ([4, 5], None),
      ([2, 3, 4], None),
      ([2, 3, 4], [2, 1, 1]),
      ([2, 3, 4], [1, 1, 2]),
  )
  def test_composition_with_nonprivate_dp_event(self, samples, counts):
    """Non-private DP event should force delta = 1 always."""
    delta = delta_calculation.delta_from_epsilon_and_samples(
        3, samples, counts, dp_accounting.NonPrivateDpEvent()
    )
    self.assertEqual(delta, 1.0)

  def test_gaussian_mc_with_gaussian_pld(self):
    """Test that composing MC and PLD matches PLD alone."""
    rng = np.random.default_rng(0)
    # Samples from PLD for Gaussian mechanism with noise multiplier 1.0.
    samples = rng.normal(loc=0.5, size=100_000)
    other_event = dp_accounting.pld.PLDAccountant(
        value_discretization_interval=1e-2
    ).compose(dp_accounting.GaussianDpEvent(1.0))
    accountant = dp_accounting.pld.PLDAccountant(
        value_discretization_interval=1e-2
    )
    accountant.compose(dp_accounting.GaussianDpEvent(1 / (2**0.5)))
    delta_1 = delta_calculation.delta_from_epsilon_and_samples(
        1.0, samples, other_event=other_event
    )
    delta_2 = accountant.get_delta(1.0)
    self.assertAlmostEqual(delta_1, delta_2, places=3)

  _FAILURE_DELTA = delta_calculation.get_base_delta(1000, 0.1)

  @parameterized.named_parameters(
      ('0_is_best', [[1] * 1000, [10] * 1000], None, None, None, (True, 0)),
      (
          'support_size_two_0_is_best',
          [[1] * 1000, [1] * 500 + [10] * 500],
          None,
          None,
          None,
          (True, 0),
      ),
      ('1_is_best', [[1] * 1000, [1] * 1000], None, None, None, (True, 1)),
      (
          'simple_0_fails',
          [[10] * 1000, [10] * 1000],
          None,
          None,
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          '1_fails_2_passes',
          [[1] * 1000, [10] * 1000, [1] * 1000],
          None,
          None,
          None,
          (True, 0),
      ),
      (
          'min_samples_used_for_base_delta',
          [[10] * 2000, [10] * 1000],
          None,
          None,
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          'simple_0_is_best_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          None,
          None,
          (True, 0),
      ),
      (
          '0_passes_positive_but_fails_negative',
          [[1] * 1000, [10] * 1000],
          None,
          [[10] * 1000, [10] * 1000],
          None,
          (False, _FAILURE_DELTA),
      ),
      (
          '0_passes_1_fails_positive',
          [[1] * 1000, [10] * 1000],
          None,
          [[1] * 1000, [1] * 1000],
          None,
          (True, 0),
      ),
      (
          '0_passes_1_fails_negative',
          [[1] * 1000, [1] * 1000],
          None,
          [[1] * 1000, [10] * 1000],
          None,
          (True, 0),
      ),
      (
          '0_passes_positive_but_fails_negative_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[10], [10]],
          [[1000], [1000]],
          (False, _FAILURE_DELTA),
      ),
      (
          '0_fails_negative_1_fails_positive_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[10], [1]],
          [[1000], [1000]],
          (False, _FAILURE_DELTA),
      ),
      (
          '1_fails_negative_with_counts',
          [[1], [10]],
          [[1000], [1000]],
          [[1], [1]],
          [[1000], [1000]],
          (True, 0),
      ),
  )
  def test_perform_calibration_from_samples(
      self,
      positive_samples,
      positive_counts,
      negative_samples,
      negative_counts,
      expected_result,
  ):
    result = delta_calculation.perform_calibration_from_samples(
        1.0,
        0.1,
        positive_samples=positive_samples,
        positive_counts=positive_counts,
        negative_samples=negative_samples,
        negative_counts=negative_counts,
    )
    self.assertEqual(result, expected_result)
    pass

  @parameterized.named_parameters(
      (
          'no_op_dp_event_no_effect_first_passes',
          [[1], [1000]],
          dp_accounting.NoOpDpEvent(),
          (True, 0),
      ),
      (
          'empty_accountant_no_effect_first_passes',
          [[1], [1000]],
          dp_accounting.pld.PLDAccountant(),
          (True, 0),
      ),
      (
          'no_op_dp_event_no_effect_all_passes',
          [[1], [1]],
          dp_accounting.NoOpDpEvent(),
          (True, 1),
      ),
      (
          'no_op_dp_event_no_effect_fails',
          [[1000], [1000]],
          dp_accounting.NoOpDpEvent(),
          (False, _FAILURE_DELTA),
      ),
      (
          'non_private_event_fails',
          [[1], [1]],
          dp_accounting.NonPrivateDpEvent(),
          (False, _FAILURE_DELTA),
      ),
      (
          'non_private_event_in_accoutant_fails',
          [[1], [1]],
          dp_accounting.pld.PLDAccountant().compose(
              dp_accounting.NonPrivateDpEvent()
          ),
          (False, _FAILURE_DELTA),
      ),
      (
          'list_of_events_determines_output',
          [[1], [1]],
          [
              dp_accounting.NoOpDpEvent(),
              dp_accounting.NonPrivateDpEvent(),
          ],
          (True, 0),
      ),
      (
          'list_of_accountants_determines_output',
          [[1], [1]],
          [
              dp_accounting.pld.PLDAccountant(),
              dp_accounting.pld.PLDAccountant().compose(
                  dp_accounting.NonPrivateDpEvent()
              ),
          ],
          (True, 0),
      ),
      (
          'list_of_events_doesnt_determine_output',
          [[1000], [1]],
          [
              dp_accounting.NoOpDpEvent(),
              dp_accounting.NonPrivateDpEvent(),
          ],
          (False, _FAILURE_DELTA),
      ),
  )
  def test_perform_calibration_from_samples_with_other_event(
      self, positive_samples, other_event, expected_result
  ):
    result = delta_calculation.perform_calibration_from_samples(
        1.0,
        0.1,
        positive_samples=positive_samples,
        positive_counts=[[1000], [1000]],
        other_event=other_event,
    )
    self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
