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
from jax_privacy import batch_selection
from jax_privacy.experimental.monte_carlo import sample_generation
import numpy as np


class SampleGenerationTest(parameterized.TestCase):

  @parameterized.parameters([
      (True, np.array([1.0]), np.array([1.0, 0.0, 1.0, 0.0])),
      (True, np.array([1.0, 0.5]), np.array([1.0, 0.5, 1.0, 0.5])),
      (False, np.array([1.0, 0.5]), np.array([0.0, 0.0, 0.0, 0.0])),
      (
          True,
          np.array([1.0, 0.5, 0.25, 0.125]),
          np.array([1.0, 0.5, 1.25, 0.625]),
      ),
      (
          False,
          np.array([1.0, 0.5, 0.25, 0.125]),
          np.array([0.0, 0.0, 0.0, 0.0]),
      ),
  ])
  def test_generate_balls_in_bins_sample_low_noise(
      self, positive_sample, c_col, first_mode
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-9
    # The distribution of samples should be evenly divided between the first
    # mode and the second mode, which is just the first mode shifted by 1
    # position. For positive_sample=False, these are the same.
    second_mode = np.zeros_like(first_mode)
    second_mode[1:] = first_mode[:-1]
    first_mode_count = 0
    for _ in range(10000):
      sample = sample_generation.generate_sample(
          sampling_scheme,
          noise_multiplier,
          c_col,
          positive_sample=positive_sample,
      )
      is_first_mode = np.allclose(sample, first_mode, atol=1e-6)
      is_second_mode = np.allclose(sample, second_mode, atol=1e-6)
      self.assertTrue(is_first_mode or is_second_mode)
      if is_first_mode:
        first_mode_count += 1
    # 6 standard deviations away from the mean.
    if positive_sample:
      self.assertGreater(first_mode_count, 4700)
      self.assertLess(first_mode_count, 5300)

  @parameterized.parameters([
      (
          True,
          np.array([1.0]),
          0.5,
          False,
          [
              np.array([1.0, 0.0, 1.0]),
              np.array([1.0, 0.0, 0.0]),
              np.array([0.0, 1.0, 0.0]),
              np.array([0.0, 0.0, 1.0]),
              np.array([0.0, 0.0, 0.0]),
          ],
          np.array([1 / 4, 1 / 4, 1 / 4, 1 / 8, 1 / 8]),
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.5,
          False,
          [
              np.array([1.0, 0.5, 1.0]),
              np.array([1.0, 0.5, 0.0]),
              np.array([0.0, 1.0, 0.5]),
              np.array([0.0, 0.0, 1.0]),
              np.array([0.0, 0.0, 0.0]),
          ],
          np.array([1 / 4, 1 / 4, 1 / 4, 1 / 8, 1 / 8]),
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.25,
          False,
          [
              np.array([1.0, 0.5, 1.0]),
              np.array([1.0, 0.5, 0.0]),
              np.array([0.0, 1.0, 0.5]),
              np.array([0.0, 0.0, 1.0]),
              np.array([0.0, 0.0, 0.0]),
          ],
          np.array([1 / 16, 3 / 16, 3 / 16, 9 / 64, 27 / 64]),
      ),
      (
          True,
          np.array([1.0, 0.5]),
          0.5,
          True,
          [
              np.array([1.0, 0.5, 1.0]),
              np.array([1.0, 0.5, 0.0]),
              np.array([0.0, 1.0, 0.5]),
              np.array([0.0, 0.0, 1.0]),
              np.array([0.0, 0.0, 0.0]),
          ],
          np.array([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6]),
      ),
      (
          False,
          np.array([1.0]),
          0.5,
          False,
          [
              np.array([0.0, 0.0, 0.0]),
          ],
          np.array([1.0]),
      ),
  ])
  def test_generate_b_min_sep_sample_low_noise(
      self,
      positive_sample,
      c_col,
      sampling_prob,
      warm_start,
      modes,
      mode_distribution,
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=sampling_prob,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
    )
    mode_counts = np.zeros(len(modes))
    noise_multiplier = 1e-9
    num_samples = 10000
    for _ in range(num_samples):
      sample = sample_generation.generate_sample(
          sampling_scheme,
          noise_multiplier,
          c_col,
          positive_sample=positive_sample,
      )
      for i, mode in enumerate(modes):
        if np.allclose(sample, mode, atol=1e-6):
          mode_counts[i] += 1
    differences = np.abs(mode_counts - num_samples * mode_distribution)
    stdev = np.sqrt(num_samples * mode_distribution * (1 - mode_distribution))
    # All within 6 standard deviations
    assert np.all(differences <= 6 * stdev)

  @parameterized.parameters([
      (
          np.array([1.0, 0.0, 1.0, 0.0]),
          np.array([1.0, 0.0]),
          0.4337808304830272,
      ),
      (
          np.array([1.0, 0.5, 1.0, 0.5]),
          np.array([1.0, 0.5]),
          0.9052974004451105,
      ),
      (
          np.array([1.0, 1.0, 1.0, 1.0]),
          np.array([1.0, 1.0, 0.5, 0.5]),
          1.5799760835798948,
      ),
  ])
  def test_compute_privacy_loss_balls_in_bins(
      self, sample, c_col, expected_privacy_loss
  ):
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1.0
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        noise_multiplier,
        c_col,
    )
    self.assertAlmostEqual(privacy_loss, expected_privacy_loss, places=6)

  @parameterized.parameters([
      (
          np.array([1.0, 1.0, 1.0]),
          np.array([1.0]),
          False,
          0.6070560625306676,
      ),
      (
          np.array([1.0, 1.0, 1.0]),
          np.array([1.0, 0.5]),
          False,
          0.9239798890121712,
      ),
      (
          np.array([1.0, 1.0, 1.0]),
          np.array([1.0, 0.5]),
          True,
          0.8329398380809252,
      ),
  ])
  def test_compute_privacy_loss_b_min_sep(
      self, sample, c_col, warm_start, expected_privacy_loss
  ):
    sampling_scheme = batch_selection.BMinSepSampling(
        sampling_prob=0.5,
        min_sep=2,
        iterations=3,
        warm_start=warm_start,
    )
    privacy_loss = sample_generation.compute_privacy_loss(
        sampling_scheme,
        sample,
        1.0,
        c_col,
    )
    self.assertAlmostEqual(privacy_loss, expected_privacy_loss, places=6)

  def test_get_privacy_loss_positive_sample(self):
    # Test that this method combines drawing a sample and computing its privacy
    # loss correctly on a low-noise example.
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    # In this setup, the privacy loss is very close to 1.25e8 for samples from
    # the first mode, and very close to 1.125e8 for samples from the second
    # mode.
    pl_samples = []
    for _ in range(10000):
      pl_samples.append(
          sample_generation.get_privacy_loss_sample(
              sampling_scheme,
              noise_multiplier,
              c_col,
              positive_sample=True,
          )
      )
    first_mode_count = sum(np.isclose(pl_samples, 1.25e8, atol=1e5))
    second_mode_count = sum(np.isclose(pl_samples, 1.125e8, atol=1e5))
    self.assertEqual(first_mode_count + second_mode_count, 10000)
    self.assertLess(first_mode_count, 5300)
    self.assertGreater(first_mode_count, 4700)

  def test_get_privacy_loss_negative_sample(self):
    # Test that this method combines drawing a sample and computing its privacy
    # loss correctly on a low-noise example.
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    # In this setup, the privacy loss is very close to 1.125e8 always.
    pl_samples = []
    for _ in range(10000):
      pl_samples.append(
          sample_generation.get_privacy_loss_sample(
              sampling_scheme,
              noise_multiplier,
              c_col,
              positive_sample=False,
          )
      )
    mode_count = sum(np.isclose(pl_samples, 1.125e8, atol=1e5))
    self.assertEqual(mode_count, 10000)

  def test_get_privacy_loss_and_sample(self):
    # Test that we can also get the sample if desired.
    sampling_scheme = batch_selection.BallsInBinsSampling(
        cycle_length=2, iterations=4
    )
    noise_multiplier = 1e-4
    c_col = np.array([1.0, 0.5])
    _, _ = sample_generation.get_privacy_loss_sample(
        sampling_scheme,
        noise_multiplier,
        c_col,
        positive_sample=True,
        also_return_sample=True,
    )


if __name__ == "__main__":
  absltest.main()
