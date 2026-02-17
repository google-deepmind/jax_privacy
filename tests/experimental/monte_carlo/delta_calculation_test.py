# coding=utf-8
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

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.experimental.monte_carlo import delta_calculation


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


if __name__ == '__main__':
  absltest.main()
