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
from jax_privacy.experimental import clipping
from jax_privacy.experimental import execution_plan
import numpy as np
import optax

NeighboringRelation = dp_accounting.NeighboringRelation
ADD_OR_REMOVE_ONE = NeighboringRelation.ADD_OR_REMOVE_ONE

DEFAULT_BANDMF_CONFIG = dict(
    epsilon=None, delta=None, noise_multiplier=1.0, num_bands=10, iterations=20
)


# TODO: Improve test coverage, including correctness of the
# privacy guarantees.
class ExecutionPlanTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(epsilon=None, delta=1e-06, noise_multiplier=1.0),
      dict(epsilon=1.0, delta=None, noise_multiplier=1.0),
      dict(epsilon=1.0, delta=1e-06, noise_multiplier=1.0),
      dict(neighboring_relation=ADD_OR_REMOVE_ONE),
      dict(
          neighboring_relation=ADD_OR_REMOVE_ONE,
          accountant=dp_accounting.pld.PLDAccountant(ADD_OR_REMOVE_ONE),
          truncated_batch_size=5,
      ),
  )
  def test_bandmf_validation(self, **overrides):
    """BandMFExecutionPlan raises at construction time for invalid configs."""
    kwargs = dict(DEFAULT_BANDMF_CONFIG)
    kwargs.update(overrides)
    with self.assertRaises(ValueError):
      execution_plan.BandMFExecutionPlan(**kwargs)

  @parameterized.parameters(
      dict(epsilon=None, delta=None, noise_multiplier=1.0),
      dict(epsilon=1.0, delta=1e-06, noise_multiplier=None),
      dict(truncated_batch_size=5, num_examples=10),
  )
  def test_bandmf_execution_plan_creation(self, **overrides):
    kwargs = dict(DEFAULT_BANDMF_CONFIG)
    kwargs.update(overrides)

    config = execution_plan.BandMFExecutionPlan(**kwargs)
    self.assertIsInstance(config, execution_plan.DPExecutionPlan)

    grad_fn = config.clipped_grad(jnp.mean)
    self.assertIsInstance(grad_fn, clipping.BoundedSensitivityCallable)

    self.assertIsInstance(config, execution_plan.DPExecutionPlan)

    batch_selection_strategy = config.batch_selection_strategy(
        shuffle=False, even_partition=False
    )
    self.assertIsInstance(
        batch_selection_strategy, batch_selection.CyclicPoissonSampling
    )
    self.assertEqual(batch_selection_strategy.sampling_prob, 1.0)
    self.assertIsInstance(
        config.noise_addition_transform(), optax.GradientTransformation,
    )
    self.assertLen(
        list(batch_selection_strategy.batch_iterator(100)), config.iterations
    )

    # TODO: b/415360727 - Add tests that the execution plan is correctly
    # configured and has the expected DP properties.

    self.assertIsInstance(config.dp_event, dp_accounting.DpEvent)
    batch_gen = batch_selection_strategy.batch_iterator(100, rng=0)
    self.assertIsInstance(next(batch_gen), np.ndarray)

  @parameterized.parameters(
      dict(epsilon=None, delta=None, noise_multiplier=1.0),
      dict(epsilon=1.0, delta=1e-06, noise_multiplier=None),
      dict(truncated_batch_size=5, num_examples=10),
  )
  def test_bandmf_raises_at_runtime(self, **overrides):
    """Should raise at runtime for valid configs but invalid overrides."""
    kwargs = dict(DEFAULT_BANDMF_CONFIG)
    kwargs.update(overrides)
    config = execution_plan.BandMFExecutionPlan(**kwargs)

    with self.assertRaises(TypeError):
      config.clipped_grad(jnp.mean, l2_clip_norm=0.0)

    with self.assertRaises(TypeError):
      config.batch_selection_strategy(sampling_prob=0.5)

    with self.assertRaises(TypeError):
      config.noise_addition_transform(privatizer_kwargs=dict(stddev=0.0001))


if __name__ == "__main__":
  absltest.main()
