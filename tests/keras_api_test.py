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

import dataclasses
import os
import types
from unittest import mock

os.environ["KERAS_BACKEND"] = "jax"
# pylint: disable=g-import-not-at-top, wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from jax_privacy import keras_api
import keras
import numpy as np
from scipy import stats
# pylint: enable=g-import-not-at-top, wrong-import-position


class KerasApiTest(parameterized.TestCase):

  def _get_params(self):
    return keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=10,
        gradient_accumulation_steps=1,
        train_steps=100,
        train_size=1000,
    )

  def test_validate_params(self):
    # Valid parameters, does not raise an error.
    valid_params = self._get_params()

    # Invalid epsilon
    with self.assertRaisesRegex(ValueError, "Epsilon .* must be positive"):
      dataclasses.replace(valid_params, epsilon=0.0)

    # Invalid delta
    with self.assertRaisesRegex(ValueError, "Delta .* must be positive"):
      dataclasses.replace(valid_params, delta=0.0)

    # Invalid batch size
    with self.assertRaisesRegex(ValueError, "Batch size .* must be positive"):
      dataclasses.replace(valid_params, batch_size=0)

    # Invalid clipping norm
    with self.assertRaisesRegex(
        ValueError, "Clipping norm .* must be positive"
    ):
      dataclasses.replace(valid_params, clipping_norm=0.0)

    # Invalid train steps
    with self.assertRaisesRegex(ValueError, "Train steps .* must be positive"):
      dataclasses.replace(valid_params, train_steps=0)

    # Invalid train size
    with self.assertRaisesRegex(ValueError, "Train size .* must be positive"):
      dataclasses.replace(valid_params, train_size=0)

    # Invalid noise multiplier
    with self.assertRaisesRegex(
        ValueError, "Noise multiplier .* must be positive"
    ):
      dataclasses.replace(valid_params, noise_multiplier=0.0)

    # Noise multiplier is too small
    with self.assertRaisesRegex(
        ValueError,
        "Value error occured while calculating epsilon",
    ):
      dataclasses.replace(valid_params, noise_multiplier=1e-10)

    # Noise multiplier exceeds privacy budget
    with self.assertRaisesRegex(
        ValueError,
        "Provided self.noise_multiplier=0.1 will lead to privacy budget exceed",
    ):
      dataclasses.replace(valid_params, noise_multiplier=0.1)

    # Gradient accumulation steps must be positive
    with self.assertRaisesRegex(
        ValueError,
        "Gradient accumulation steps 0 must be positive",
    ):
      dataclasses.replace(valid_params, gradient_accumulation_steps=0)

    with self.assertRaisesRegex(
        ValueError,
        "sampling_prob must be in \\(0, 1\\]",
    ):
      dataclasses.replace(valid_params, sampling_prob=1.1)

    with self.assertRaisesRegex(
        ValueError,
        "batch_size \\(2000\\) must be <= train_size \\(1000\\)",
    ):
      dataclasses.replace(valid_params, batch_size=2000)

    with self.assertRaisesRegex(
        ValueError,
        "padding_multiple must be a positive integer or None",
    ):
      dataclasses.replace(valid_params, padding_multiple=0)

    with self.assertRaisesRegex(
        ValueError,
        "padding_multiple must be divisible by microbatch_size",
    ):
      dataclasses.replace(
          valid_params,
          microbatch_size=7,
          padding_multiple=10,
      )

  def test_effective_batch_size(self):
    params1 = dataclasses.replace(self._get_params(), batch_size=5)
    self.assertEqual(params1.effective_batch_size, 5)

    params2 = dataclasses.replace(params1, gradient_accumulation_steps=10)
    self.assertEqual(params2.effective_batch_size, 50)

  def test_poisson_sampling_prob(self):
    params = self._get_params()
    self.assertAlmostEqual(params.poisson_sampling_prob, 0.01)
    params = dataclasses.replace(params, sampling_prob=0.1)
    self.assertEqual(params.poisson_sampling_prob, 0.1)

  def test_dp_params_calculates_noise_multiplier(self):
    params = keras_api.DPKerasConfig(
        noise_multiplier=None,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=10,
        gradient_accumulation_steps=1,
        train_steps=100,
        train_size=1000,
    )

    updated_params = params.update_with_calibrated_noise_multiplier()

    self.assertGreater(updated_params.noise_multiplier, 0.0)

  def test_add_dp_sgd_attributes(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        batch_size=10,
        gradient_accumulation_steps=1,
        clipping_norm=1.0,
        train_steps=20,
        train_size=500,
        noise_multiplier=10.0,
        microbatch_size=1,
    )

    keras_api._add_dp_sgd_attributes(model, params)

    self.assertTrue(hasattr(model, "_dp_params"))
    self.assertEqual(model._dp_params, params)
    self.assertEqual(model._dp_noise_multiplier, params.noise_multiplier)

  def test_get_noise_multiplier_uses_config_value(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = dataclasses.replace(self._get_params(), noise_multiplier=2.5)
    private_model = keras_api.make_private(model, params)

    self.assertEqual(private_model.get_noise_multiplier(), 2.5)

  def test_get_noise_multiplier_calibrates_once(self):
    model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
    params = self._get_params()
    private_model = keras_api.make_private(model, params)

    noise_multiplier = private_model.get_noise_multiplier()
    self.assertIsNotNone(noise_multiplier)
    self.assertGreater(noise_multiplier, 0.0)
    self.assertEqual(private_model._dp_noise_multiplier, noise_multiplier)
    self.assertEqual(private_model.get_noise_multiplier(), noise_multiplier)

  @parameterized.named_parameters(
      ("no_rescale_no_clip", 100.0, 1, False, [-10.0, -20.0]),
      ("no_rescale_clip", 1.0, 1, False, [-0.44721362, -0.89442724]),
      # rescale=true: clipped_grads = grads / max(clipping_norm, norm(grads))
      ("rescale_big_clipping_norm", 100.0, 1, True, [-0.1, -0.2]),
      ("rescale_clipping_norm_1", 1.0, 1, True, [-0.44721362, -0.89442724]),
      ("rescale_clipping_norm_2", 2.0, 1, True, [-0.44721362, -0.89442724]),
      ("batch_size_5", 100.0, 5, False, [-10.0, -20.0]),
  )
  def test_clipped_grads(
      self,
      clipping_norm: float,
      batch_size: int,
      rescale_to_unit_norm: bool,
      expected_grads: list[float],
  ):
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=20,
        train_size=500,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )
    # turn off noise for testing
    object.__setattr__(dp_params, "noise_multiplier", 0.0)

    # The function is (a0*x0+a1*x1-4)^2, where a0, a1 = 3, -2, x0, x1 = 1, 2.
    # We compute gradient respect a0, a1, so
    # grad(f) = (2*x0*(a0*x0+a1*x1-4), 2*x1*(a0*x0+a1*x1-4)) =
    # (2*(3-4-4), 4*(3-4-4)) = (-10, -20).
    trainable_variables = [jnp.array([3.0, -2.0])]
    x = jnp.array([[1.0, 2.0] for _ in range(batch_size)])
    y = jnp.array([4.0] * batch_size)

    noise_rng = jax.random.PRNGKey(0)
    non_trainable_variables = [noise_rng]
    state = (trainable_variables, non_trainable_variables, [], [])
    data = (x, y, None)

    (loss, _), grads = keras_api._noised_clipped_grads(
        _compute_mse_loss_and_updates_fn,
        dp_params,
        state,
        data,
    )

    self.assertEqual(loss, jnp.array([25.0]))
    chex.assert_trees_all_close(grads[0], jnp.array(expected_grads))

  @parameterized.parameters((1.1, 200, 1), (10, 50, 10), (5, 100, 20))
  def test_noise_distribution(
      self, epsilon: float, clipping_norm: float, batch_size: int
  ):
    dp_params = keras_api.DPKerasConfig(
        epsilon=epsilon,
        delta=1e-5,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=500,
        train_size=500,
        rescale_to_unit_norm=False,
    ).update_with_calibrated_noise_multiplier()

    # The function is (a0*x0+a1*x1-4)^2, where a0, a1 = 3, -2, x0, x1 = 1, 2.
    # We compute gradient respect a0, a1, so
    # grad(f) = (2*x0*(a0*x0+a1*x1-4), 2*x1*(a0*x0+a1*x1-4)) =
    # (2*(3-4-4), 4*(3-4-4)) = (-10, -20).
    trainable_variables = [jnp.array([3.0, -2.0])]
    x = jnp.array([[1.0, 2.0]] * batch_size)
    y = jnp.array([4.0] * batch_size)

    # Generate sample.
    sample = []
    for _ in range(100):  # ~5 seconds
      noise_rng = jax.random.PRNGKey(keras_api._get_random_int64())
      non_trainable_variables = [noise_rng]
      state = (trainable_variables, non_trainable_variables, [], [])
      data = (x, y, None)

      _, grads = keras_api._noised_clipped_grads(
          _compute_mse_loss_and_updates_fn,
          dp_params,
          state,
          data,
      )
      sample.append(np.array(grads[0]))

    # Check that noised gradients aligned with expected distribution. The
    # gradient is 2 element vector, and the first element ~N(-10, stddev**2),
    # the second ~N(-20, stddev**2).
    sample = np.stack(sample)
    stddev = dp_params.noise_multiplier * clipping_norm / batch_size
    self._check_distribution(sample[:, 0], -10, stddev)
    self._check_distribution(sample[:, 1], -20, stddev)

  def test_validate_optimizer_mismatched_gradient_accumulation_steps(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = dataclasses.replace(
        self._get_params(), gradient_accumulation_steps=10
    )
    model = keras_api.make_private(model, dp_params)
    optimizer = keras.optimizers.Adam(gradient_accumulation_steps=5)
    model.compile(optimizer=optimizer)
    with self.assertRaisesRegex(
        ValueError, "optimizer.gradient_accumulation_steps = 5 must be equal to"
    ):
      keras_api._validate_optimizer(model, dp_params)

  def test_dp_training_e2e_work(self):
    np.random.seed(42)
    train_size = 200
    batch_size = 100
    epochs = 5
    train_steps = 10  # 5 * (200 / 100)
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, epochs=epochs, batch_size=batch_size)  # pylint: disable=not-callable
    self.assertAlmostEqual(model.evaluate(x, y), 2, delta=2)

  def test_dp_training_exceeds_privacy_budget_raises_error(self):
    train_size = 200
    batch_size = 100
    train_steps = 28
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    # fit is wrapped and does arguments parsing therefore below we try setting
    # different arguments in different ways (in args and in kwargs).
    # optimizer steps per epoch = 200 / 100 = 2
    # batch_size is in args
    model.fit(x, y, batch_size, epochs=2)  # optimizer steps = 2 * 2 = 4  # pylint: disable=not-callable
    model.fit(  # pylint: disable=not-callable
        x, y, epochs=2, batch_size=batch_size
    )  # optimizer steps = 4 + 2 * 2 = 8
    # initial_epoch is set
    model.fit(  # pylint: disable=not-callable
        x, y, batch_size, initial_epoch=4, epochs=5
    )  # optimizer steps = 8 + 1 * 2 = 10
    # steps_per_epoch is set
    model.fit(  # pylint: disable=not-callable
        x, y, batch_size, epochs=3, steps_per_epoch=2
    )  # optimizer steps = 10 + 3 * 2 = 16
    with self.assertRaisesRegex(
        RuntimeError,
        "you will run out of privacy budget",
    ):
      # cannot be performed because 16 + 7 * 2 = 30 > 28
      model.fit(x, y, epochs=7, batch_size=batch_size)  # pylint: disable=not-callable

  def test_fit_with_missing_args(self):
    # Arrange.
    train_size = 64
    train_steps = 15
    dp_params = keras_api.DPKerasConfig(
        batch_size=32,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=train_steps,
        train_size=train_size,
    )
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    original_fit_fn = model.fit

    # Signature of fit function of Gemma model in Keras.
    # For example it doesn't have epochs argument.
    def _fit_fn_with_missing_args(  # pylint: disable=too-many-positional-arguments
        _,  # unused self
        x=None,
        y=None,
        batch_size=None,
        sample_weight=None,
        validation_data=None,
        validation_split=None,
        **kwargs
    ):
      return original_fit_fn(  # pylint: disable=not-callable
          x=x,
          y=y,
          batch_size=batch_size,
          sample_weight=sample_weight,
          validation_data=validation_data,
          validation_split=validation_split,
          **kwargs
      )

    model.fit = types.MethodType(_fit_fn_with_missing_args, model)
    model = keras_api.make_private(model, dp_params)
    model.compile()

    # Act & assert.
    # train_steps = epochs (=1) * train_size (=64) / batch_size (=32)= 2
    model.fit(x, y, batch_size=32)  # optimizer steps = 1 * 2 = 2  # pylint: disable=not-callable
    with self.assertRaisesRegex(
        RuntimeError,
        "you will run out of privacy budget",
    ):
      # cannot be performed because 2 + 7 * 2 = 16 > 15
      model.fit(x, y, epochs=7)  # pylint: disable=not-callable

  def test_fit_raises_error_if_dp_params_not_aligned_with_fit_args(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=200,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "The batch size in the DP parameters is not equal to the batch size"
        " passed to fit()",
    ):
      model.fit(batch_size=101)  # pylint: disable=not-callable

  def test_fit_raises_error_for_non_random_access_training_data(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=200,
    )
    model = keras_api.make_private(model, dp_params)

    def data_generator():
      while True:
        yield np.zeros((100, 4)), np.zeros((100,))

    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "random access",
    ):
      model.fit(data_generator())  # pylint: disable=not-callable

  def test_fit_raises_error_when_train_size_does_not_match_data(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    x, y = np.zeros((101, 4)), np.zeros((101,))
    dp_params = keras_api.DPKerasConfig(
        batch_size=100,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=28,
        train_size=200,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "train_size",
    ):
      model.fit(x, y, batch_size=100)  # pylint: disable=not-callable

  def test_fit_raises_error_for_validation_split(self):
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    train_size = 100
    x, y = np.zeros((train_size, 4)), np.zeros((train_size,))
    dp_params = keras_api.DPKerasConfig(
        batch_size=50,
        gradient_accumulation_steps=1,
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        train_steps=2,
        train_size=train_size,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile()
    with self.assertRaisesRegex(
        ValueError,
        "validation_split is not supported",
    ):
      model.fit(  # pylint: disable=not-callable
          x,
          y,
          batch_size=50,
          validation_split=0.1,
      )

  def test_poisson_data_iterator_pads_and_masks_dummy_examples(self):
    train_size = 16
    x = np.ones((train_size, 2), dtype=np.float32)
    y = np.ones((train_size,), dtype=np.float32)
    params = keras_api.DPKerasConfig(
        epsilon=1.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        train_steps=4,
        train_size=train_size,
        sampling_prob=0.05,
        padding_multiple=8,
    )
    iterator = keras_api._create_poisson_data_iterator(
        x=x,
        y=y,
        sample_weight=None,
        dp_params=params,
        rng=np.random.default_rng(0),
        iterations=1,
    )
    batch_x, batch_y, batch_sample_weight = next(iterator)
    self.assertEqual(batch_x.shape[0], 8)
    self.assertEqual(batch_y.shape[0], 8)
    self.assertEqual(batch_sample_weight.shape[0], 8)
    self.assertTrue(
        set(np.unique(batch_sample_weight).tolist()).issubset({0.0, 1.0})
    )
    padding_indices = np.where(batch_sample_weight == 0)[0]
    if padding_indices.size:
      np.testing.assert_allclose(batch_x[padding_indices], 0.0)

  def test_padding_examples_are_zeroed_in_clipped_grad(self):
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=100.0,
        batch_size=3,
        gradient_accumulation_steps=1,
        train_steps=1,
        train_size=3,
    )
    object.__setattr__(dp_params, "noise_multiplier", 0.0)
    trainable_variables = [jnp.array([3.0, -2.0])]
    noise_rng = jax.random.PRNGKey(0)
    non_trainable_variables = [noise_rng]
    state = (trainable_variables, non_trainable_variables, [], [])
    real_x = jnp.array([[1.0, 2.0], [1.0, 2.0]])
    real_y = jnp.array([4.0, 4.0])
    dummy_x = jnp.array([[1000.0, 1000.0]])
    dummy_y = jnp.array([1000.0])

    _, grads_without_padding = keras_api._noised_clipped_grads(
        _compute_mse_loss_and_updates_fn,
        dp_params,
        state,
        (real_x, real_y, jnp.ones((2,))),
    )
    _, grads_with_padding = keras_api._noised_clipped_grads(
        _compute_mse_loss_and_updates_fn,
        dp_params,
        state,
        (
            jnp.concatenate([real_x, dummy_x], axis=0),
            jnp.concatenate([real_y, dummy_y], axis=0),
            jnp.array([1.0, 1.0, 0.0]),
        ),
        is_padding_example=jnp.array([False, False, True]),
    )
    chex.assert_trees_all_close(grads_without_padding, grads_with_padding)

  def test_dp_training_with_dict_inputs(self):
    train_size = 32
    batch_size = 8
    x = {
        "features": np.random.uniform(0, 1, (train_size, 4)).astype(np.float32)
    }
    y = np.random.uniform(0, 1, train_size).astype(np.float32)
    inputs = keras.Input(shape=(4,), name="features")
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=train_size // batch_size,
        train_size=train_size,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")
    history = model.fit(  # pylint: disable=not-callable
        x,
        y,
        batch_size=batch_size,
        epochs=1,
    )
    self.assertIn("loss", history.history)

  def test_train_step_call_noised_clipped_grads(self):
    train_size = 200
    batch_size = 100
    epochs = 1
    train_steps = 2  # 1 * (200 / 100)
    x, y = np.random.uniform(0, 1, (train_size, 4)), np.random.uniform(
        0, 1, train_size
    )
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(1)])
    dp_params = keras_api.DPKerasConfig(
        epsilon=1.1,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=100,
        gradient_accumulation_steps=1,
        train_steps=train_steps,
        train_size=train_size,
    )
    model = keras_api.make_private(model, dp_params)
    model.compile(loss="mse", optimizer="adam")

    with mock.patch.object(
        keras_api,
        "_noised_clipped_grads",
        wraps=keras_api._noised_clipped_grads,
    ) as mock_noised_clipped_grads:
      model.fit(x, y, epochs=epochs, batch_size=batch_size)  # pylint: disable=not-callable
      self.assertGreaterEqual(mock_noised_clipped_grads.call_count, 1)

  def _check_distribution(
      self, sample: np.ndarray, mean: float, stddev: float
  ) -> None:
    """Verifies the hypothesis that the sample comes from N(mean, stddev**2)."""
    size = len(sample)
    sample_expected_distribution = np.random.normal(mean, stddev, size=size)
    pvalue = stats.ks_2samp(sample_expected_distribution, sample).pvalue
    self.assertGreater(pvalue, 1e-5)  # expected flakiness 1e-5


def _compute_mse_loss_and_updates_fn(  # pylint: disable=too-many-positional-arguments
    params,
    non_trainable_variables,
    metrics_variables,
    x,
    y,
    sample_weight,
    training,
    optimizer_variables,
):
  """Compute mse loss and updates for a simple model.

  This function has the same signature as JaxTrainer.compute_loss_and_updates_fn
  in order to replace in testing of gradient computation.

  Args:
    params: The parameters of the model.
    non_trainable_variables: The non-trainable variables of the model. Not used.
    metrics_variables: The metrics variables of the model. Not used.
    x: The input data.
    y: The target data.
    sample_weight: The sample weight.
    training: Whether the model is in training mode. Not used.
    optimizer_variables: The optimizer variables of the model. Not used.

  Returns:
    (loss, aux), where
      loss is the loss value.
      aux is a tuple (y_pred, non_trainable_variables, metrics_variables).
  """
  del (
      non_trainable_variables,
      metrics_variables,
      training,
      optimizer_variables,
  )
  y_pred = jnp.sum(x * params[0], axis=1)
  if sample_weight is not None:
    loss_value = jnp.mean(sample_weight * (y_pred - y) ** 2)
  else:
    loss_value = jnp.mean((y_pred - y) ** 2)
  aux = (loss_value, y_pred, [], [])
  return loss_value, aux


if __name__ == "__main__":
  absltest.main()
