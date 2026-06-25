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
import jax
import jax.numpy as jnp
from jax_privacy import execution_plan
from jax_privacy.experimental import training
import numpy as np
import optax


def _quadratic_loss(params, batch, prng):
  """Per-example quadratic loss with has_aux=True convention."""
  del prng
  loss = jnp.mean((params - batch) ** 2)
  return loss, {'loss': loss}


def _make_plan(
    iterations,
    noise_multiplier=1.0,
    expected_participations=None,
    performance_flags=None,
):
  """Creates a simple BandMF execution plan for testing."""
  if expected_participations is None:
    expected_participations = iterations
  config = execution_plan.BandMFConfig.default(
      num_bands=1,
      iterations=iterations,
      noise_multiplier=noise_multiplier,
      expected_participations=expected_participations,
  )
  return config.make(performance_flags=performance_flags)


class DPTrainerTest(parameterized.TestCase):
  """Tests for the DPTrainer class."""

  def test_basic_training_runs(self):
    """Train loop completes and returns a valid TrainingState."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    plan = _make_plan(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), 3)

  def test_params_change_after_training(self):
    """Parameters should change from initial values after training."""
    params = jnp.array([10.0, 10.0])
    dataset = np.array([[0.0, 0.0], [0.0, 0.0]])
    plan = _make_plan(iterations=5)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=42)

    self.assertFalse(jnp.allclose(state.params, params))

  def test_callback_invoked(self):
    """Callback should be invoked once per training step."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0]])
    iterations = 3
    plan = _make_plan(iterations=iterations)
    optimizer = optax.sgd(0.01)
    callback_log = []

    def callback(step, state, aux):
      callback_log.append((int(step), float(state.params[0])))
      self.assertIsInstance(aux.grad_norms, jax.Array)
      self.assertIsInstance(aux.values, jax.Array)
      self.assertIsNotNone(aux.aux)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    trainer.fit(dataset, params, callback=callback, rng=0)

    self.assertLen(callback_log, iterations)
    self.assertEqual([s for s, _ in callback_log], [1, 2, 3])

  def test_padding_multiple(self):
    """Training should work with padding_multiple set."""
    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0], [2.0]])
    plan = _make_plan(iterations=2)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
        padding_multiple=4,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 2)

  def test_default_padding_multiple(self):
    """Default padding_multiple should be 32 to prevent JIT recompilation."""
    plan = _make_plan(iterations=1)
    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optax.sgd(0.01),
    )
    self.assertEqual(trainer.padding_multiple, 32)

  def test_zero_iterations_config_raises(self):
    """BandMFConfig requires iterations >= 1."""
    with self.assertRaises(Exception):
      _make_plan(iterations=0)

  def test_single_iteration(self):
    """Training with a single iteration should work correctly."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 0.0]])
    plan = _make_plan(iterations=1)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 1)

  def test_loss_fn_traced_once_with_equal_batches(self):
    """loss_fn should only be traced once with constant batches."""
    trace_count = [0]

    def counting_loss(params, batch, prng):
      del prng
      trace_count[0] += 1
      loss = jnp.mean((params - batch) ** 2)
      return loss, {}

    params = jnp.array([1.0])
    dataset = np.array([[0.0], [1.0]])
    plan = _make_plan(iterations=3)
    optimizer = optax.sgd(0.01)

    jax.clear_caches()

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=counting_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 3)
    self.assertLess(trace_count[0], 3 * 2)

  def test_train_step_callable_directly(self):
    """train_step should be directly callable outside of fit()."""
    params = jnp.array([5.0, 5.0])
    plan = _make_plan(iterations=2, noise_multiplier=0.0)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )

    state = training.TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        noise_state=plan.noise_addition_transform.init(params),
    )

    batch = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    is_padding = jnp.array([False, False])
    loss_rng = jax.random.key(0)

    new_state, _ = trainer.train_step(
        state, batch, is_padding, loss_rng=loss_rng
    )

    self.assertEqual(int(new_state.step), 1)
    self.assertFalse(jnp.allclose(new_state.params, params))

  def test_train_step_jit_compilable(self):
    """train_step should be JIT-compilable."""
    params = jnp.array([5.0])
    plan = _make_plan(iterations=1, noise_multiplier=0.0)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )

    state = training.TrainingState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        noise_state=plan.noise_addition_transform.init(params),
    )

    batch = jnp.array([[1.0], [0.0]])
    is_padding = jnp.array([False, False])
    loss_rng = jax.random.key(0)

    jit_step = jax.jit(trainer.train_step)
    new_state, _ = jit_step(state, batch, is_padding, loss_rng=loss_rng)

    self.assertEqual(int(new_state.step), 1)


class DPTrainerEdgeCasesTest(parameterized.TestCase):
  """Edge case tests for the DPTrainer class."""

  def test_epsilon_zero_high_noise(self):
    """Near-zero epsilon (very high noise) should run without error."""
    params = jnp.array([1.0, 2.0])
    dataset = np.array([[0.0, 0.0], [1.0, 1.0]])
    plan = _make_plan(iterations=2, noise_multiplier=1e6)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 2)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))

  def test_epsilon_inf_no_noise(self):
    """noise_multiplier=0 should behave like non-private SGD."""
    params = jnp.array([5.0])
    dataset = np.array([[0.0], [0.0]])
    plan = _make_plan(iterations=3, noise_multiplier=0.0)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 3)
    self.assertLess(
        float(jnp.abs(state.params[0])),
        float(jnp.abs(params[0])),
    )

  def test_single_example_dataset(self):
    """Training on a single example should work correctly."""
    params = jnp.array([5.0])
    dataset = np.array([[0.0]])
    plan = _make_plan(iterations=2, noise_multiplier=0.0)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 2)
    self.assertTrue(jnp.all(jnp.isfinite(state.params)))

  def test_dict_dataset(self):
    """Training should work with dict-structured datasets."""
    params = jnp.array([1.0])
    dataset = {'x': np.array([[0.0], [1.0], [2.0]])}

    def dict_loss(params, batch, prng):
      del prng
      loss = jnp.mean((params - batch['x']) ** 2)
      return loss, {}

    plan = _make_plan(iterations=2)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=dict_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 2)

  def test_bfloat16_params_preserved(self):
    """bfloat16 params with float32 plan should return bfloat16."""
    params = jnp.array([5.0, 5.0], dtype=jnp.bfloat16)
    dataset = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    flags = execution_plan.PerformanceFlags(dtype=np.float32)
    plan = _make_plan(
        iterations=2,
        noise_multiplier=0.0,
        performance_flags=flags,
    )
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan,
        loss_fn=_quadratic_loss,
        optimizer=optimizer,
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(state.params.dtype, jnp.bfloat16)


if __name__ == '__main__':
  absltest.main()
