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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from jax_privacy.experimental import accounting

MAKE_EVENT_FNS = (
    lambda nm: accounting.dpsgd_event(nm, 128, sampling_prob=0.01),
    functools.partial(
        accounting.amplified_bandmf_event,
        iterations=128,
        num_bands=16,
        sampling_prob=0.01,
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
        _make_test_case(event_fn, RDP_ACCOUNTANT) for event_fn in MAKE_EVENT_FNS
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


if __name__ == "__main__":
  absltest.main()
