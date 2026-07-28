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
import dp_accounting
import jax.numpy as jnp
from jax_privacy import batch_selection
from jax_privacy import execution_plan
import numpy as np
import optax

BandMFConfig = execution_plan.BandMFConfig


# pylint: disable=g-bad-todo
# TODO: Improve test coverage, including correctness of the
# privacy guarantees.
class ExecutionPlanTest(parameterized.TestCase):

  @parameterized.parameters(
      {"strategy": np.array([])},
      {"truncated_batch_size": 5, "num_examples": None},
  )
  def test_bandmf_validation(self, **kwargs):
    default_kwargs = {
        "strategy": np.linspace(1, 0, 10),
        "iterations": 20,
        "expected_participations": 2,
        "noise_multiplier": 1.0,
    }
    default_kwargs.update(kwargs)
    with self.assertRaises(ValueError):
      BandMFConfig(**default_kwargs)

  @parameterized.parameters(
      {
          "noise_multiplier": 1.0,
      },
      {
          "noise_multiplier": 1.0,
          "truncated_batch_size": 5,
          "num_examples": 10,
      },
  )
  def test_bandmf_execution_plan_creation(self, **privacy_kwargs):
    iterations = 20
    config = BandMFConfig.default(
        num_bands=10,
        iterations=iterations,
        expected_participations=iterations / 10,
        **privacy_kwargs,
    )

    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
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

    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)
    batch_gen = plan.batch_selection_strategy.batch_iterator(100, rng=0)
    self.assertIsInstance(next(batch_gen), np.ndarray)

  def test_bandmf_calibrate(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
    ).calibrate(epsilon=1.0, delta=1e-06)

    self.assertIsNotNone(config.noise_multiplier)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

  def test_uncalibrated_make_raises_error(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
    )
    with self.assertRaises(ValueError):
      config.make()

  def test_make_with_default_performance_flags(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
        noise_multiplier=1.0,
    )
    plan = config.make()
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)

  def test_make_with_custom_performance_flags(self):
    config = BandMFConfig.default(
        num_bands=10,
        iterations=20,
        expected_participations=2,
        noise_multiplier=1.0,
    )
    flags = execution_plan.PerformanceFlags(
        dtype=np.float64,
        noise_seed=42,
        microbatch_size=4,
    )
    plan = config.make(flags)
    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)

  def test_rmse_requires_calibration(self):
    config = BandMFConfig.default(
        num_bands=1,
        iterations=10,
        expected_participations=10,
    )
    with self.assertRaises(ValueError):
      _ = config.rmse

  def test_rmse_decreases_with_participations(self):
    config1 = BandMFConfig.default(
        num_bands=2,
        iterations=16,
        expected_participations=4,
    ).calibrate(epsilon=1.0, delta=1e-06)
    config2 = BandMFConfig.default(
        num_bands=2,
        iterations=16,
        expected_participations=2,
    ).calibrate(epsilon=1.0, delta=1e-06)
    self.assertLess(config1.rmse, config2.rmse)

  def test_non_private_config(self):
    """Tests that NonPrivateConfig creates a valid non-private plan."""
    iterations = 20
    batch_size = 5
    config = execution_plan.NonPrivateConfig(
        iterations=iterations,
        batch_size=batch_size,
    )
    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.FixedBatchSampling
    )
    self.assertEqual(plan.batch_selection_strategy.batch_size, batch_size)
    self.assertEqual(plan.batch_selection_strategy.iterations, iterations)
    self.assertIsInstance(
        plan.noise_addition_transform, optax.GradientTransformation
    )
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

    # Verify that the privatizer acts as identity (no-op)
    dummy_grads = {"w": np.ones((2, 2))}
    opt_state = plan.noise_addition_transform.init(dummy_grads)
    updates, _ = plan.noise_addition_transform.update(dummy_grads, opt_state)
    np.testing.assert_equal(updates, dummy_grads)


class DpsgdConfigTest(parameterized.TestCase):
  """Tests for Poisson and fixed-size DP-SGD execution plan configs."""

  def test_dpsgd_validation(self):
    with self.assertRaises(ValueError):
      execution_plan.DpsgdConfig(
          iterations=10,
          expected_batch_size=100,
          num_examples=50,
          noise_multiplier=1.0,
      )

  def test_dpsgd_uncalibrated_make_raises(self):
    config = execution_plan.DpsgdConfig(
        iterations=20,
        expected_batch_size=8,
        num_examples=100,
    )
    with self.assertRaises(ValueError):
      config.make()

  def test_dpsgd_execution_plan_creation(self):
    iterations = 20
    expected_batch_size = 8
    num_examples = 100
    config = execution_plan.DpsgdConfig(
        iterations=iterations,
        expected_batch_size=expected_batch_size,
        num_examples=num_examples,
        noise_multiplier=1.0,
        l2_clip_norm=1.0,
        rescale_to_unit_norm=True,
    )
    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.CyclicPoissonSampling
    )
    self.assertAlmostEqual(
        plan.batch_selection_strategy.sampling_prob,
        expected_batch_size / num_examples,
    )
    self.assertEqual(plan.batch_selection_strategy.cycle_length, 1)
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    self.assertIsInstance(plan.dp_event, dp_accounting.SelfComposedDpEvent)
    self.assertLen(
        list(plan.batch_selection_strategy.batch_iterator(num_examples)),
        iterations,
    )

    grad_fn = plan.clipped_grad(lambda params, x: (params - x).mean())
    self.assertAlmostEqual(
        grad_fn.sensitivity(plan.neighboring_relation),
        1.0 / expected_batch_size,
    )

  def test_dpsgd_calibrate_meets_budget(self):
    epsilon = 2.0
    delta = 1e-5
    config = execution_plan.DpsgdConfig(
        iterations=50,
        expected_batch_size=16,
        num_examples=500,
    ).calibrate(epsilon=epsilon, delta=delta)

    self.assertIsNotNone(config.noise_multiplier)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    accountant = dp_accounting.pld.PLDAccountant(plan.neighboring_relation)
    realized_epsilon = accountant.compose(plan.dp_event).get_epsilon(
        target_delta=delta
    )
    self.assertLessEqual(realized_epsilon, epsilon + 1e-3)

  def test_dpsgd_noise_matches_sensitivity_calibration(self):
    """Empirical noise stddev matches noise_multiplier * sensitivity."""
    expected_batch_size = 10
    config = execution_plan.DpsgdConfig(
        iterations=5,
        expected_batch_size=expected_batch_size,
        num_examples=100,
        noise_multiplier=2.5,
        l2_clip_norm=1.0,
        rescale_to_unit_norm=True,
        normalize_by=expected_batch_size,
    )
    plan = config.make(execution_plan.PerformanceFlags(noise_seed=0))
    sensitivity = plan.clipped_grad(lambda: None).sensitivity(
        plan.neighboring_relation
    )
    expected_stddev = config.noise_multiplier * sensitivity

    grads = {"w": jnp.zeros((2048,))}
    state = plan.noise_addition_transform.init(grads)
    noisy, _ = plan.noise_addition_transform.update(grads, state)
    empirical_std = float(jnp.std(noisy["w"]))
    # Large vector: std estimate should be close to the calibrated stddev.
    self.assertAlmostEqual(empirical_std, expected_stddev, delta=0.15)

  def test_dpsgd_truncated_plan(self):
    config = execution_plan.DpsgdConfig(
        iterations=10,
        expected_batch_size=8,
        num_examples=100,
        truncated_batch_size=12,
        noise_multiplier=1.0,
    )
    plan = config.make()
    self.assertEqual(plan.batch_selection_strategy.truncated_batch_size, 12)
    for batch in plan.batch_selection_strategy.batch_iterator(100, rng=0):
      self.assertLessEqual(batch.size, 12)

  def test_fixed_size_dpsgd_execution_plan(self):
    iterations = 15
    batch_size = 8
    num_examples = 64
    config = execution_plan.FixedSizeDpsgdConfig(
        iterations=iterations,
        batch_size=batch_size,
        num_examples=num_examples,
        noise_multiplier=1.5,
    )
    plan = config.make()
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.FixedBatchSampling
    )
    self.assertEqual(plan.batch_selection_strategy.batch_size, batch_size)
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.REPLACE_ONE,
    )
    batches = list(
        plan.batch_selection_strategy.batch_iterator(num_examples, rng=1)
    )
    self.assertLen(batches, iterations)
    for batch in batches:
      self.assertLen(batch, batch_size)

    grad_fn = plan.clipped_grad(lambda params, x: (params - x).mean())
    # REPLACE_ONE doubles sensitivity vs add/remove.
    self.assertAlmostEqual(
        grad_fn.sensitivity(plan.neighboring_relation),
        2.0 / batch_size,
    )

  def test_fixed_size_dpsgd_calibrate(self):
    config = execution_plan.FixedSizeDpsgdConfig(
        iterations=30,
        batch_size=16,
        num_examples=200,
    ).calibrate(epsilon=3.0, delta=1e-5)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

  def test_fixed_size_validation(self):
    with self.assertRaises(ValueError):
      execution_plan.FixedSizeDpsgdConfig(
          iterations=10,
          batch_size=100,
          num_examples=50,
          replace=False,
          noise_multiplier=1.0,
      )


if __name__ == "__main__":
  absltest.main()
