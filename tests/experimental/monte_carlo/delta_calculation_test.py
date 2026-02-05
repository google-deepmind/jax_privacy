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
from jax_privacy.experimental.monte_carlo import delta_calculation


class DeltaCalculationTest(parameterized.TestCase):

  @parameterized.parameters(0.01, 0.05, 0.1)
  def test_overall_delta(self, base_delta):
    # Check that overall delta is decreasing in num_samples, and always in
    # (base_delta, 1].
    overall_delta_10000 = delta_calculation.get_overall_delta(10000, base_delta)
    overall_delta_10001 = delta_calculation.get_overall_delta(10001, base_delta)
    self.assertLessEqual(overall_delta_10000, 1.0)
    self.assertLess(overall_delta_10001, overall_delta_10000)
    self.assertLess(base_delta, overall_delta_10001)

  @parameterized.parameters((10000, 0.01), (3000, 0.05), (1000, 0.1))
  def test_base_delta(self, num_samples, target_delta):
    base_delta = delta_calculation.get_base_delta(num_samples, target_delta)
    overall_delta = delta_calculation.get_overall_delta(num_samples, base_delta)
    self.assertLessEqual(overall_delta, target_delta)
    self.assertAlmostEqual(overall_delta, target_delta, places=5)

  @parameterized.parameters(
      (100, 0.01),
      (20, 0.05),
      (10, 0.1),
  )
  def test_num_samples_too_small(self, num_samples, target_delta):
    with self.assertRaises(ValueError):
      delta_calculation.get_base_delta(num_samples, target_delta)


if __name__ == '__main__':
  absltest.main()
