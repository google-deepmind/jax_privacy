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
import dp_accounting
import jax.numpy as jnp
from jax_privacy.experimental import batch_selection
from jax_privacy.experimental import execution_plan
from jax_privacy.experimental import gradient_clipping
from jax_privacy.noise_addition import gradient_privatizer
import numpy as np


# TODO: Improve test coverage, including correctness of the
# privacy guarantees.
class ExecutionPlanTest(parameterized.TestCase):

  @parameterized.parameters(
      {"epsilon": None, "delta": None, "noise_multiplier": None},
      {"epsilon": 1.0, "delta": 1e-06, "noise_multiplier": 2.0},
      {"num_bands": 0},
  )
  def test_bandmf_validation(self, **kwargs):
    default_kwargs = {
        "num_examples": 100,
        "num_bands": 10,
        "iterations": 20,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 1.0,
    }
    default_kwargs.update(kwargs)
    with self.assertRaises(ValueError):
      execution_plan.BandMFExecutionPlanConfig(**default_kwargs)

  def test_bandmf_execution_plan_creation(self):

    iterations = 20
    config = execution_plan.BandMFExecutionPlanConfig(
        num_examples=100,
        num_bands=10,
        iterations=iterations,
        epsilon=None,
        delta=None,
        noise_multiplier=1.0,
        shuffle=False,
        use_fixed_size_groups=False,
    )

    gradient_fn = gradient_clipping.clipped_grad(jnp.mean, l2_clip_norm=1.0)
    plan = config.make(gradient_fn)

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    # Assert that the batch selection strategy is CyclicPoissonSampling with
    # sampling_prob = 1.0, which is equivalent to shuffling /
    # (k, b)-participation.
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.CyclicPoissonSampling
    )
    self.assertEqual(plan.batch_selection_strategy.sampling_prob, 1.0)
    self.assertIsInstance(
        plan.noise_addition_transform,
        gradient_privatizer.GradientPrivatizer,
    )
    self.assertLen(
        list(plan.batch_selection_strategy.batch_iterator()), iterations
    )

    # TODO: b/415360727 - Add tests that the execution plan is correctly
    # configured and has the expected DP properties.

    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)
    batch_gen = plan.batch_selection_strategy.batch_iterator(rng=0)
    self.assertIsInstance(next(batch_gen), np.ndarray)


if __name__ == "__main__":
  absltest.main()
