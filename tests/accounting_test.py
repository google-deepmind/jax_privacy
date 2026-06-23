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

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy import accounting

MAKE_EVENT_FNS = (
    lambda nm: accounting.dpsgd_event(nm, 128, sampling_prob=0.01),
    functools.partial(
        accounting.amplified_bandmf_event,
        iterations=128,
        num_bands=16,
        sampling_prob=0.01,
    ),
)

RDP_ONLY_EVENT_FNS = (
    functools.partial(
        accounting.dpsgd_event,
        iterations=128,
        sampling_prob=0.01,
        use_zcdp=True,
    ),
)

PLD_EVENT_FNS = (
    functools.partial(
        accounting.truncated_dpsgd_event,
        iterations=128,
        sampling_prob=0.01,
        num_examples=1000,
        truncated_batch_size=16,
    ),
    functools.partial(
        accounting.truncated_amplified_bandmf_event,
        iterations=128,
        num_bands=16,
        sampling_prob=0.01,
        largest_group_size=1000,
        truncated_batch_size=16,
    ),
)

RDP_REPLACE_EVENT_FNS = (
    functools.partial(
        accounting.fixed_dpsgd_event,
        iterations=128,
        dataset_size=1000,
        batch_size=16,
        replace=False,
    ),
    functools.partial(
        accounting.fixed_dpsgd_event,
        iterations=128,
        dataset_size=1000,
        batch_size=16,
        replace=False,
        use_zcdp=True,
    ),
)

PLD_ACCOUNTANT = functools.partial(
    dp_accounting.pld.PLDAccountant, value_discretization_interval=1e-3
)
RDP_ACCOUNTANT = dp_accounting.rdp.RdpAccountant  # pylint: disable=invalid-name
RDP_REPLACE_ACCOUNTANT = functools.partial(
    dp_accounting.rdp.RdpAccountant,
    neighboring_relation=(
        dp_accounting.privacy_accountant.NeighboringRelation.REPLACE_ONE
    ),
)


def _make_test_case(event_fn, accountant):
  return {"event_fn": event_fn, "accountant": accountant}


TEST_CASES = (
    tuple(
        _make_test_case(ev, PLD_ACCOUNTANT)
        for ev in MAKE_EVENT_FNS + PLD_EVENT_FNS
    )
    + tuple(
        _make_test_case(event_fn, RDP_ACCOUNTANT)
        for event_fn in MAKE_EVENT_FNS + RDP_ONLY_EVENT_FNS
    )
    + tuple(
        _make_test_case(event_fn, RDP_REPLACE_ACCOUNTANT)
        for event_fn in RDP_REPLACE_EVENT_FNS
    )
)


class AccountingTest(parameterized.TestCase):

  @parameterized.parameters(*TEST_CASES)
  def test_get_epsilon(self, event_fn, accountant):
    event = event_fn(1.0)
    epsilon = accountant().compose(event).get_epsilon(target_delta=1e-6)
    self.assertGreaterEqual(epsilon, 0)

  @parameterized.parameters(*TEST_CASES)
  def test_calibration(self, event_fn, accountant):
    nm = dp_accounting.calibrate_dp_mechanism(
        accountant,
        event_fn,
        target_epsilon=1.0,
        target_delta=1e-6,
        bracket_interval=dp_accounting.ExplicitBracketInterval(0.1, 20.0),
    )
    eps = accountant().compose(event_fn(nm)).get_epsilon(target_delta=1e-6)
    self.assertAlmostEqual(eps, 1.0, places=4)

  def test_fixed_dpsgd_event_replace_true_type(self):
    event = accounting.fixed_dpsgd_event(
        1.0,
        5,
        dataset_size=10,
        batch_size=3,
        replace=True,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertIsInstance(
        event.event, dp_accounting.dp_event.SampledWithReplacementDpEvent
    )

  def test_fixed_dpsgd_event_use_zcdp_structure(self):
    noise_multiplier = 3.0
    iterations = 5
    dataset_size = 10
    batch_size = 3
    event = accounting.fixed_dpsgd_event(
        noise_multiplier,
        iterations,
        dataset_size=dataset_size,
        batch_size=batch_size,
        use_zcdp=True,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertEqual(event.count, iterations)
    self.assertIsInstance(
        event.event, dp_accounting.dp_event.SampledWithoutReplacementDpEvent
    )
    inner = event.event.event
    self.assertIsInstance(inner, dp_accounting.dp_event.ZCDpEvent)
    self.assertAlmostEqual(inner.rho, 0.5 / noise_multiplier**2)

  def test_dpsgd_event_use_zcdp_structure(self):
    noise_multiplier = 3.0
    iterations = 10
    sampling_prob = 0.01
    event = accounting.dpsgd_event(
        noise_multiplier,
        iterations,
        sampling_prob=sampling_prob,
        use_zcdp=True,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertEqual(event.count, iterations)
    self.assertIsInstance(
        event.event, dp_accounting.dp_event.PoissonSampledDpEvent
    )
    self.assertAlmostEqual(event.event.sampling_probability, sampling_prob)
    inner = event.event.event
    self.assertIsInstance(inner, dp_accounting.dp_event.ZCDpEvent)
    self.assertAlmostEqual(inner.rho, 0.5 / noise_multiplier**2)

  def test_dpsgd_event_zero_noise_does_not_crash(self):
    """noise_multiplier=0 is permitted and must not raise in either branch.

    `_validate_args` only rejects negative noise multipliers, so 0.0 is a valid
    input denoting infinite privacy loss. The continuous branch already handles
    it via GaussianDpEvent(0); the zCDP branch must likewise not raise a
    ZeroDivisionError from 0.5 / noise_multiplier**2 and should report rho=inf.
    """
    continuous = accounting.dpsgd_event(0.0, 10, sampling_prob=0.01)
    self.assertIsInstance(
        continuous.event.event, dp_accounting.dp_event.GaussianDpEvent
    )
    self.assertEqual(continuous.event.event.noise_multiplier, 0.0)

    discrete = accounting.dpsgd_event(
        0.0, 10, sampling_prob=0.01, use_zcdp=True
    )
    inner = discrete.event.event
    self.assertIsInstance(inner, dp_accounting.dp_event.ZCDpEvent)
    self.assertTrue(math.isinf(inner.rho))

  def test_use_zcdp_matches_gaussian_privacy(self):
    """use_zcdp=True & default should give the same epsilon with RDP."""
    noise_multiplier = 5.0
    iterations = 128
    sampling_prob = 0.01
    target_delta = 1e-6

    continuous_event = accounting.dpsgd_event(
        noise_multiplier, iterations, sampling_prob=sampling_prob
    )
    discrete_event = accounting.dpsgd_event(
        noise_multiplier,
        iterations,
        sampling_prob=sampling_prob,
        use_zcdp=True,
    )

    accountant_continuous = dp_accounting.rdp.RdpAccountant()
    eps_continuous = accountant_continuous.compose(
        continuous_event
    ).get_epsilon(target_delta=target_delta)

    accountant_discrete = dp_accounting.rdp.RdpAccountant()
    eps_discrete = accountant_discrete.compose(discrete_event).get_epsilon(
        target_delta=target_delta
    )

    # The discrete Gaussian with rho=0.5/sigma^2 should give the same RDP
    # as the continuous Gaussian with the same sigma.
    self.assertAlmostEqual(eps_continuous, eps_discrete, places=4)

  # -- Structural-argument validation ----------------------------------------
  # num_bands flows into math.ceil(iterations / num_bands); the truncated sizes
  # flow into TruncatedSubsampledGaussianDpEvent. These were previously
  # unvalidated: num_bands=0 raised an opaque ZeroDivisionError, num_bands<0
  # raised a misleading "iterations=..." error, and negative sizes were silently
  # accepted and produced a nonsensical event.

  @parameterized.parameters(0, -1, -16)
  def test_amplified_bandmf_event_rejects_nonpositive_num_bands(
      self, num_bands
  ):
    with self.assertRaisesRegex(ValueError, rf"num_bands={num_bands} > 0"):
      accounting.amplified_bandmf_event(
          1.0, 128, num_bands=num_bands, sampling_prob=0.01
      )

  @parameterized.parameters(0, -1, -16)
  def test_truncated_amplified_bandmf_rejects_nonpositive_num_bands(
      self, num_bands
  ):
    with self.assertRaisesRegex(ValueError, rf"num_bands={num_bands} > 0"):
      accounting.truncated_amplified_bandmf_event(
          1.0,
          128,
          num_bands=num_bands,
          sampling_prob=0.01,
          largest_group_size=1000,
          truncated_batch_size=16,
      )

  @parameterized.parameters(
      dict(num_examples=-1, truncated_batch_size=16),
      dict(num_examples=1000, truncated_batch_size=-1),
      dict(num_examples=-5, truncated_batch_size=-3),
  )
  def test_truncated_dpsgd_event_rejects_negative_sizes(
      self, num_examples, truncated_batch_size
  ):
    with self.assertRaisesRegex(ValueError, r">= 0"):
      accounting.truncated_dpsgd_event(
          1.0,
          10,
          sampling_prob=0.1,
          num_examples=num_examples,
          truncated_batch_size=truncated_batch_size,
      )

  def test_truncated_amplified_bandmf_rejects_negative_sizes(self):
    # largest_group_size / truncated_batch_size are forwarded to
    # truncated_dpsgd_event and validated there.
    with self.assertRaisesRegex(ValueError, r">= 0"):
      accounting.truncated_amplified_bandmf_event(
          1.0,
          128,
          num_bands=16,
          sampling_prob=0.01,
          largest_group_size=-1,
          truncated_batch_size=16,
      )

  def test_amplified_bandmf_event_valid_num_bands_unchanged(self):
    # Regression: positive num_bands is unaffected. The mechanism runs
    # rounds = ceil(iterations / num_bands) DP-SGD steps.
    event = accounting.amplified_bandmf_event(
        1.0, 128, num_bands=16, sampling_prob=0.01
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertEqual(event.count, 8)  # ceil(128 / 16)

  def test_truncated_amplified_bandmf_valid_args_unchanged(self):
    event = accounting.truncated_amplified_bandmf_event(
        1.0,
        128,
        num_bands=16,
        sampling_prob=0.01,
        largest_group_size=1000,
        truncated_batch_size=16,
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertEqual(event.count, 8)  # ceil(128 / 16)

  def test_truncated_dpsgd_event_valid_args_unchanged(self):
    # Regression: non-negative (incl. zero) sizes still build the event.
    event = accounting.truncated_dpsgd_event(
        1.0, 10, sampling_prob=0.1, num_examples=1000, truncated_batch_size=16
    )
    self.assertIsInstance(event, dp_accounting.dp_event.SelfComposedDpEvent)
    self.assertEqual(event.count, 10)


if __name__ == "__main__":
  absltest.main()
