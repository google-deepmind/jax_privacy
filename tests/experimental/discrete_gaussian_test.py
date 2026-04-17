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

import time

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy.experimental import discrete_gaussian
import numpy as np
import scipy.stats


class SampleDiscreteGaussianTest(parameterized.TestCase):

  def test_output_shape(self):
    """Output array has the requested number of samples."""
    rng = np.random.default_rng(0)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=1.0, size=100
    )
    self.assertEqual(samples.shape, (100,))

  def test_output_dtype_is_integer(self):
    """Output values are integers."""
    rng = np.random.default_rng(0)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=1.0, size=50
    )
    self.assertTrue(np.issubdtype(samples.dtype, np.integer))

  def test_sigma_zero_returns_zeros(self):
    """When sigma=0 the distribution is degenerate at 0."""
    rng = np.random.default_rng(0)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=0.0, size=10
    )
    np.testing.assert_array_equal(samples, np.zeros(10, dtype=np.int64))

  def test_negative_sigma_raises(self):
    """Negative sigma is invalid."""
    rng = np.random.default_rng(0)
    with self.assertRaises(ValueError):
      discrete_gaussian.sample_discrete_gaussian(rng, sigma=-1.0, size=1)

  def test_zero_size_raises(self):
    """size must be positive."""
    rng = np.random.default_rng(0)
    with self.assertRaises(ValueError):
      discrete_gaussian.sample_discrete_gaussian(rng, sigma=1.0, size=0)

  def test_negative_size_raises(self):
    """Negative size is invalid."""
    rng = np.random.default_rng(0)
    with self.assertRaises(ValueError):
      discrete_gaussian.sample_discrete_gaussian(rng, sigma=1.0, size=-5)

  def test_size_one(self):
    """size=1 produces an array of size 1."""
    rng = np.random.default_rng(0)
    samples = discrete_gaussian.sample_discrete_gaussian(rng, sigma=1.0, size=1)
    self.assertEqual(samples.shape, (1,))

  def test_deterministic_with_seeded_rng(self):
    """Passing the same RNG seed produces identical results."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    samples1 = discrete_gaussian.sample_discrete_gaussian(
        rng1, sigma=2.0, size=200
    )
    samples2 = discrete_gaussian.sample_discrete_gaussian(
        rng2, sigma=2.0, size=200
    )
    np.testing.assert_array_equal(samples1, samples2)

  def test_different_seeds_differ(self):
    """Different RNG seeds produce different results (with high probability)."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(999)
    samples1 = discrete_gaussian.sample_discrete_gaussian(
        rng1, sigma=2.0, size=200
    )
    samples2 = discrete_gaussian.sample_discrete_gaussian(
        rng2, sigma=2.0, size=200
    )
    self.assertFalse(np.array_equal(samples1, samples2))

  @parameterized.parameters(0.1, 1.0, 4.0, 25.0)
  def test_mean_is_approximately_zero(self, sigma):
    """The sample mean should be close to 0."""
    rng = np.random.default_rng(12345)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=sigma, size=10_000
    )
    # Allow ±5 standard errors.
    se = sigma / np.sqrt(len(samples))
    self.assertAlmostEqual(np.mean(samples), 0.0, delta=5 * se)

  @parameterized.parameters(0.1, 1.0, 4.0, 25.0)
  def test_variance_is_less_than_sigma2(self, sigma):
    """The discrete Gaussian variance is strictly less than sigma2."""
    rng = np.random.default_rng(54321)
    n = 100_000
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=sigma, size=n
    )
    # The variance of the discrete Gaussian is strictly < sigma2.
    # The sample variance has standard error ~ sqrt(2) * sigma2 / sqrt(n).
    # We allow up to 5 standard errors above sigma2 to avoid flakiness.
    sample_var = np.var(samples, ddof=0)
    tolerance = 7 * sigma**2 / np.sqrt(n)
    self.assertLess(sample_var, sigma**2 + tolerance)
    if sigma >= 1.0:
      self.assertGreater(sample_var, 0.0)

  @parameterized.parameters(0.1, 1.0, 4.0, 25.0)
  def test_variance_is_greater_than_lower_bound(self, sigma):
    """The discrete Gaussian variance is greater than a lower bound."""
    # Lower bound is Var(X) >= 1/(exp(1/sigma**2)-1) for all sigma>0.
    # See Corollary 24 in https://arxiv.org/abs/2004.00010
    rng = np.random.default_rng(54321)
    n = 100_000
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=sigma, size=n
    )
    sample_var = np.var(samples, ddof=0)
    lower_bound = 1 / np.expm1(1 / sigma**2)
    tolerance = 57 * sigma**2 / np.sqrt(n)
    self.assertLess(lower_bound - tolerance, sample_var)
    if sigma >= 1.0:
      self.assertGreater(sample_var, 0.0)

  def test_small_sigma_concentrates_near_zero(self):
    """For very small sigma, almost all mass is on 0."""
    rng = np.random.default_rng(111)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=0.1, size=1000
    )
    fraction_zero = np.mean(samples == 0)
    self.assertGreater(fraction_zero, 0.99)

  def test_large_sigma_has_spread(self):
    """For large sigma, the samples are spread out."""
    rng = np.random.default_rng(222)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=10.0, size=1000
    )
    # With sigma2=100, std ~ 10, so most samples should be in [-30, 30]
    # but definitely not all zero.
    self.assertGreater(np.std(samples), 1.0)
    self.assertGreater(np.max(np.abs(samples)), 0)

  def test_oversample_correctness(self):
    """Setting oversample does not affect correctness."""
    rng = np.random.default_rng(333)
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=2.0, size=100, oversample=10
    )
    self.assertEqual(samples.shape, (100,))
    self.assertTrue(np.issubdtype(samples.dtype, np.integer))

  @parameterized.parameters(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000)
  def test_kl_divergence(self, sigma):
    """Use G-test to check that the distribution is correct."""
    rng = np.random.default_rng(54321 + int(sigma * 10))
    if rng.uniform() < 0.5:  # want to test both settings of oversample
      oversample = None
    else:
      oversample = 1000
    n = 10_000_000
    samples = discrete_gaussian.sample_discrete_gaussian(
        rng, sigma=sigma, size=n, oversample=oversample
    )
    # Compute KL(samples || discrete_gaussian(sigma2))
    unnormalized_kl = 0
    values, counts = np.unique(samples, return_counts=True)
    for x, count in zip(values, counts):
      # kl_divergence += count/n * np.log(count / n / p(x))
      # p(x) = exp(-x**2/2/sigma**2)/ normalizing_constant
      unnormalized_kl += count * (np.log(count) + x**2 / 2 / sigma**2)
    big_n = 100  # number of terms to truncate the infinite sum at
    y = np.arange(-big_n, big_n + 1)
    if sigma**2 < 0.5 / np.pi:
      normalizing_constant = np.sum(np.exp(-(y**2) / 2 / sigma**2))
    else:  # Use Poisson summation formula for faster convergence.
      normalizing_constant = np.sqrt(2 * np.pi * sigma**2) * np.sum(
          np.exp(-(2 * np.pi**2 * sigma**2) * y**2)
      )
    kl_divergence = (
        unnormalized_kl / n - np.log(n) + np.log(normalizing_constant)
    )
    # Check that the KL divergence is close to 0.
    # G-test should be O(1/n), but constant depends on size of the support.
    tolerance = 10 * (sigma + 1) / n
    self.assertLess(kl_divergence, tolerance)
    self.assertGreaterEqual(kl_divergence, 0.0)  # KL>=0 always.

  def test_performance(self):
    """Sampling 10M values should complete within 10 seconds."""
    rng = np.random.default_rng(555)
    start = time.monotonic()
    discrete_gaussian.sample_discrete_gaussian(rng, sigma=2.0, size=10_000_000)
    elapsed = time.monotonic() - start
    self.assertLess(elapsed, 10.0, f"Sampling 1M values took {elapsed:.2f}s")
    # This test took 2.33 seconds when I tried it.
    # Failing this test is not a big deal. E.g. it could fail due to using a
    # slower PRNG that is more secure. It's included just to flag if a change
    # makes the implementation significantly slower.

  @parameterized.parameters(0.1, 0.422, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0)
  def test_laplace_scale_parameter(self, sigma):
    """laplace_scale should be set to ensure >60% acceptance rate."""
    n = 1000
    laplace_scale, _ = discrete_gaussian._get_sampling_parameters(sigma, n)
    y = np.arange(-100, 101)  # truncated infinite sum
    if sigma**2 < 0.5 / np.pi:
      s = np.sum(np.exp(-(y**2) / 2 / sigma**2))
    else:
      s = np.sqrt(2 * np.pi * sigma**2) * np.sum(
          np.exp(-(2 * np.pi**2 * sigma**2) * y**2)
      )
    r = -np.expm1(-1 / laplace_scale) / (1 + np.exp(-1 / laplace_scale))
    accept_prob = s * np.exp(-(sigma**2) / laplace_scale**2 / 2) * r
    self.assertGreater(accept_prob, 0.6)

  def test_oversample_parameter(self):
    """oversample should be set to ensure sample enough >95% of the time."""
    size = 0
    for _ in range(700):  # Try lots of different values of size
      size = size + max(1, size // 50)
      _, oversample = discrete_gaussian._get_sampling_parameters(1.0, size)
      # Rejection sampling has acceptance probability >60%
      # Thus we get Binomial(oversample, 0.6) samples.
      # We want the probability of getting at least size samples to be >95%.
      p = scipy.stats.binom.cdf(size - 1, oversample, 0.6)
      self.assertLess(p, 0.05)


if __name__ == "__main__":
  absltest.main()
