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
  """Tests for the unified DpsgdConfig (Poisson and fixed-size)."""

  def test_dpsgd_uncalibrated_make_raises(self):
    config = execution_plan.DpsgdConfig(
        iterations=20,
        expected_participations=2.0,
    )
    with self.assertRaises(ValueError):
      config.make()

  def test_poisson_execution_plan_creation(self):
    iterations = 20
    expected_participations = 1.6
    expected_batch_size = 8
    config = execution_plan.DpsgdConfig(
        iterations=iterations,
        expected_participations=expected_participations,
        noise_multiplier=1.0,
        l2_clip_norm=1.0,
        rescale_to_unit_norm=True,
        normalize_by=expected_batch_size,
    )
    plan = config.make()

    self.assertIsInstance(plan, execution_plan.DPExecutionPlan)
    self.assertIsInstance(
        plan.batch_selection_strategy, batch_selection.CyclicPoissonSampling
    )
    self.assertAlmostEqual(
        plan.batch_selection_strategy.sampling_prob,
        expected_participations / iterations,
    )
    self.assertEqual(plan.batch_selection_strategy.cycle_length, 1)
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    self.assertIsInstance(plan.dp_event, dp_accounting.SelfComposedDpEvent)
    self.assertLen(
        list(plan.batch_selection_strategy.batch_iterator(100)),
        iterations,
    )

    grad_fn = plan.clipped_grad(lambda params, x: (params - x).mean())
    self.assertAlmostEqual(
        grad_fn.sensitivity(plan.neighboring_relation),
        1.0 / expected_batch_size,
    )

  def test_poisson_calibrate_meets_budget(self):
    epsilon = 2.0
    delta = 1e-5
    config = execution_plan.DpsgdConfig(
        iterations=50,
        expected_participations=1.6,
        normalize_by=16,
    ).calibrate(epsilon=epsilon, delta=delta)

    self.assertIsNotNone(config.noise_multiplier)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    accountant = dp_accounting.pld.PLDAccountant(plan.neighboring_relation)
    realized_epsilon = accountant.compose(plan.dp_event).get_epsilon(
        target_delta=delta
    )
    self.assertLessEqual(realized_epsilon, epsilon + 1e-3)

  def test_poisson_noise_matches_l2_norm_bound_calibration(self):
    """Empirical noise stddev matches noise_multiplier * l2_norm_bound."""
    expected_batch_size = 10
    config = execution_plan.DpsgdConfig(
        iterations=5,
        expected_participations=0.5,
        noise_multiplier=2.5,
        l2_clip_norm=1.0,
        rescale_to_unit_norm=True,
        normalize_by=expected_batch_size,
    )
    plan = config.make(execution_plan.PerformanceFlags(noise_seed=0))
    l2_norm_bound = plan.clipped_grad(lambda: None).l2_norm_bound
    expected_stddev = config.noise_multiplier * l2_norm_bound

    grads = {"w": jnp.zeros((2048,))}
    state = plan.noise_addition_transform.init(grads)
    noisy, _ = plan.noise_addition_transform.update(grads, state)
    empirical_std = float(jnp.std(noisy["w"]))
    self.assertAlmostEqual(empirical_std, expected_stddev, delta=0.15)

  def test_truncated_poisson_requires_num_examples(self):
    with self.assertRaises(ValueError):
      execution_plan.DpsgdConfig(
          iterations=10,
          expected_participations=1.0,
          truncated_batch_size=12,
          noise_multiplier=1.0,
      )

  def test_truncated_poisson_plan(self):
    config = execution_plan.DpsgdConfig(
        iterations=10,
        expected_participations=0.8,
        num_examples=100,
        truncated_batch_size=12,
        noise_multiplier=1.0,
        normalize_by=8,
    )
    plan = config.make()
    self.assertEqual(plan.batch_selection_strategy.truncated_batch_size, 12)
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.REPLACE_SPECIAL,
    )
    for batch in plan.batch_selection_strategy.batch_iterator(100, rng=0):
      self.assertLessEqual(batch.size, 12)

  def test_fixed_size_requires_num_examples(self):
    with self.assertRaises(ValueError):
      execution_plan.DpsgdConfig(
          iterations=10,
          expected_participations=2.0,
          batch_selection=execution_plan.DpsgdBatchSelection.FIXED,
          noise_multiplier=1.0,
      )

  def test_fixed_size_execution_plan(self):
    iterations = 15
    batch_size = 8
    num_examples = 64
    config = execution_plan.DpsgdConfig(
        iterations=iterations,
        expected_participations=iterations * batch_size / num_examples,
        batch_selection=execution_plan.DpsgdBatchSelection.FIXED,
        num_examples=num_examples,
        noise_multiplier=1.5,
        normalize_by=batch_size,
    )
    self.assertEqual(config.expected_batch_size, batch_size)
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
    self.assertAlmostEqual(
        grad_fn.sensitivity(plan.neighboring_relation),
        2.0 / batch_size,
    )

  def test_fixed_size_calibrate(self):
    config = execution_plan.DpsgdConfig(
        iterations=30,
        expected_participations=2.4,
        batch_selection=execution_plan.DpsgdBatchSelection.FIXED,
        num_examples=200,
        normalize_by=16,
    ).calibrate(epsilon=3.0, delta=1e-5)
    self.assertEqual(config.expected_batch_size, 16)
    self.assertGreater(config.noise_multiplier, 0)
    plan = config.make()
    self.assertIsInstance(plan.dp_event, dp_accounting.DpEvent)

  def test_fixed_size_noise_matches_l2_norm_bound_not_sensitivity(self):
    """Under REPLACE_ONE, noise scales by l2_norm_bound, not sensitivity().

    dp_accounting's noise_multiplier is relative to l2_norm_bound. Using
    sensitivity() (== 2 * l2_norm_bound under REPLACE_ONE) would over-noise.
    """
    batch_size = 8
    num_examples = 64
    noise_multiplier = 3.0
    config = execution_plan.DpsgdConfig(
        iterations=10,
        expected_participations=10 * batch_size / num_examples,
        batch_selection=execution_plan.DpsgdBatchSelection.FIXED,
        num_examples=num_examples,
        noise_multiplier=noise_multiplier,
        l2_clip_norm=1.0,
        rescale_to_unit_norm=True,
        normalize_by=batch_size,
    )
    plan = config.make(execution_plan.PerformanceFlags(noise_seed=1))
    self.assertEqual(
        plan.neighboring_relation,
        dp_accounting.NeighboringRelation.REPLACE_ONE,
    )
    grad_fn = plan.clipped_grad(lambda: None)
    l2_norm_bound = grad_fn.l2_norm_bound
    sensitivity = grad_fn.sensitivity(plan.neighboring_relation)
    self.assertAlmostEqual(sensitivity, 2.0 * l2_norm_bound)
    expected_stddev = noise_multiplier * l2_norm_bound
    wrong_stddev = noise_multiplier * sensitivity

    grads = {"w": jnp.zeros((4096,))}
    state = plan.noise_addition_transform.init(grads)
    noisy, _ = plan.noise_addition_transform.update(grads, state)
    empirical_std = float(jnp.std(noisy["w"]))
    self.assertAlmostEqual(empirical_std, expected_stddev, delta=0.12)
    # Must not match the 2x-inflated REPLACE_ONE sensitivity scale.
    self.assertGreater(
        abs(empirical_std - wrong_stddev), abs(empirical_std - expected_stddev)
    )

  def test_fixed_size_requires_integer_inferred_batch_size(self):
    with self.assertRaises(ValueError):
      execution_plan.DpsgdConfig(
          iterations=10,
          expected_participations=1.5,
          batch_selection=execution_plan.DpsgdBatchSelection.FIXED,
          num_examples=50,
          noise_multiplier=1.0,
      )


if __name__ == "__main__":
  absltest.main()
