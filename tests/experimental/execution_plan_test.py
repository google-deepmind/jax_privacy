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
from jax_privacy import batch_selection
from jax_privacy import clipping
from jax_privacy.experimental import execution_plan
import numpy as np
import optax


# pylint: disable=g-bad-todo
# TODO: Improve test coverage, including correctness of the
# privacy guarantees.
class ExecutionPlanTest(parameterized.TestCase):

  @parameterized.parameters(
      {"epsilon": None, "delta": None, "noise_multiplier": None},
      {"epsilon": 1.0, "delta": 1e-06, "noise_multiplier": 2.0},
      {"num_bands": 0},
      {"truncated_batch_size": 5, "num_examples": None},
  )
  def test_bandmf_validation(self, **kwargs):
    default_kwargs = {
        "num_bands": 10,
        "iterations": 20,
        "epsilon": None,
        "delta": None,
        "noise_multiplier": 1.0,
    }
    default_kwargs.update(kwargs)
    with self.assertRaises(ValueError):
      execution_plan.BandMFExecutionPlanConfig(**default_kwargs)

  @parameterized.parameters(
      {
          "epsilon": None,
          "delta": None,
          "noise_multiplier": 1.0,
      },
      {"epsilon": 1.0, "delta": 1e-06, "noise_multiplier": None},
      {
          "epsilon": None,
          "delta": None,
          "noise_multiplier": 1.0,
          "truncated_batch_size": 5,
          "num_examples": 10,
      },
  )
  def test_bandmf_execution_plan_creation(self, **privacy_kwargs):

    iterations = 20
    config = execution_plan.BandMFExecutionPlanConfig(
        num_bands=10, iterations=iterations, **privacy_kwargs
    )

    gradient_fn = clipping.clipped_grad(jnp.mean, l2_clip_norm=1.0)
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
        optax.GradientTransformation,
    )
    self.assertLen(
        list(plan.batch_selection_strategy.batch_iterator(100)), iterations
    )

    # TODO: b/415360727 - Add tests that the execution plan is correctly
    # configured and has the expected DP properties.

    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)
    batch_gen = plan.batch_selection_strategy.batch_iterator(100, rng=0)
    self.assertIsInstance(next(batch_gen), np.ndarray)

  @parameterized.parameters(
      {
          "epsilon": None,
          "delta": 1e-06,
          "noise_multiplier": 1.0,
      },
      {
          "epsilon": 1.0,
          "delta": None,
          "noise_multiplier": 1.0,
      },
      {
          "epsilon": 1.0,
          "delta": 1e-06,
          "noise_multiplier": None,
          "neighboring_relation": (
              dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
          ),
      },
      {
          "epsilon": 1.0,
          "delta": 1e-06,
          "noise_multiplier": None,
          "neighboring_relation": (
              dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
          ),
          "accountant": dp_accounting.pld.PLDAccountant(
              dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
          ),
          "truncated_batch_size": 5,
      },
  )
  def test_bandmf_execution_plan_creation_raises_error(self, **privacy_kwargs):
    with self.assertRaises(ValueError):
      execution_plan.BandMFExecutionPlanConfig(
          num_bands=10,
          iterations=20,
          **privacy_kwargs,
      )


if __name__ == "__main__":
  absltest.main()
