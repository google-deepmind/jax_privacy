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

"""Tests for the Keras-to-DPTrainer helpers in the DP-SAPF example."""

import dataclasses
import importlib.util
import math
import os
import pathlib
from unittest import mock

os.environ["KERAS_BACKEND"] = "jax"
# pylint: disable=g-import-not-at-top, wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import accounting
from jax_privacy import training
import keras
import numpy as np
# pylint: enable=g-import-not-at-top, wrong-import-position


# Examples are not installed as a package. Resolve the script from this file
# so test collection does not depend on the working directory or PYTHONPATH.
_EXAMPLE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "dpsapf_validate.py"
)
_SPEC = importlib.util.spec_from_file_location("dpsapf_validate", _EXAMPLE_PATH)
if _SPEC is None or _SPEC.loader is None:
  raise ImportError("Unable to load the DP-SAPF validation example.")
dpsapf_validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dpsapf_validate)


class DpsapfValidateTest(parameterized.TestCase):

  def test_materialize_preserves_full_population_in_host_arrays(self):
    examples = np.arange(5, dtype=np.int32)
    dataset = mock.Mock(spec=["batch"])
    dataset.batch.return_value = [examples[:2], examples[2:4], examples[4:]]

    def preprocess(chunk):
      tokens = jnp.stack([chunk, chunk + 10], axis=-1)
      return {"token_ids": tokens}, tokens + 1, None

    preprocessor = mock.Mock(side_effect=preprocess)
    materialized = dpsapf_validate._materialize_training_data(
        dataset, preprocessor, num_examples=5, chunk_size=2
    )

    dataset.batch.assert_called_once_with(2, drop_remainder=False)
    self.assertEqual(preprocessor.call_count, 3)
    self.assertEqual(preprocessor.call_args.args[0].shape[0], 1)
    expected_tokens = np.stack([examples, examples + 10], axis=-1)
    np.testing.assert_array_equal(
        materialized["x"]["token_ids"], expected_tokens
    )
    np.testing.assert_array_equal(materialized["y"], expected_tokens + 1)
    np.testing.assert_array_equal(
        materialized["sw"], np.ones_like(expected_tokens, dtype=np.float32)
    )
    for leaf in jax.tree.leaves(materialized):
      self.assertIsInstance(leaf, np.ndarray)

  def test_loss_uses_sample_weights_and_inference_mode(self):
    inputs = keras.Input(shape=(2, 3))
    frozen = keras.layers.Dense(
        3, use_bias=False, kernel_initializer="identity", trainable=False
    )(inputs)
    dropped = keras.layers.Dropout(0.75, seed=7)(frozen)
    logits = keras.layers.Dense(2, use_bias=False)(dropped)
    model = keras.Model(inputs, logits)
    model.trainable_variables[0].assign(
        np.array([[0.5, -0.2], [-0.1, 0.3], [0.2, 0.4]], np.float32)
    )
    batch = {
        "x": jnp.array([
            [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
            [[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
        ]),
        "y": jnp.array([[0, 1], [1, 0]]),
        "sw": jnp.array([[1.0, 0.0], [0.5, 2.0]]),
    }
    params = tuple(variable.value for variable in model.trainable_variables)
    frozen_before = [
        np.array(variable.value) for variable in model.non_trainable_variables
    ]
    loss_fn = dpsapf_validate._make_keras_loss(model)

    loss, aux = loss_fn(params, batch, jax.random.key(0))
    loss_other_key, _ = loss_fn(params, batch, jax.random.key(1))

    inference_logits = model(batch["x"], training=False)
    log_probs = jax.nn.log_softmax(inference_logits)
    selected_log_probs = jnp.take_along_axis(
        log_probs, batch["y"][..., None], axis=-1
    )[..., 0]
    expected_loss = -jnp.mean(selected_log_probs * batch["sw"])
    self.assertEqual(loss.shape, ())
    self.assertEqual(aux, ())
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(loss_other_key, expected_loss, rtol=1e-6)
    zero_weight_loss, _ = loss_fn(
        params,
        {**batch, "sw": jnp.zeros_like(batch["sw"])},
        jax.random.key(2),
    )
    self.assertEqual(float(zero_weight_loss), 0.0)
    for before, variable in zip(
        frozen_before, model.non_trainable_variables, strict=True
    ):
      np.testing.assert_array_equal(variable, before)

  def test_calibration_accounts_for_probe_and_effective_batch(self):
    probe_event = accounting.dpsgd_event(
        noise_multiplier=3.0, iterations=1, sampling_prob=0.5
    )
    epsilon, delta = 3.0, 1e-5
    with mock.patch.object(
        dp_accounting,
        "calibrate_dp_mechanism",
        wraps=dp_accounting.calibrate_dp_mechanism,
    ) as calibrate:
      config = dpsapf_validate._calibrate_training_config(
          train_size=10,
          batch_size=2,
          gradient_accumulation_steps=2,
          epochs=2,
          clipping_norm=0.3,
          probe_dp_event=probe_event,
          epsilon=epsilon,
          delta=delta,
      )

    calibrate.assert_called_once()
    self.assertIsInstance(config, training.BandMFConfig)
    self.assertEqual(config.iterations, 4)
    self.assertAlmostEqual(config.expected_participations, 1.6)
    self.assertEqual(config.normalize_by, 4)
    self.assertEqual(config.l2_clip_norm, 0.3)
    self.assertFalse(config.rescale_to_unit_norm)
    np.testing.assert_array_equal(config.strategy, [1.0])
    self.assertGreater(config.noise_multiplier, 0.0)

    plan = config.make()
    self.assertEqual(plan.batch_selection_strategy.iterations, 4)
    self.assertAlmostEqual(plan.batch_selection_strategy.sampling_prob, 0.4)
    self.assertIsInstance(plan.dp_event, dp_accounting.SelfComposedDpEvent)
    self.assertEqual(plan.dp_event.count, 4)
    self.assertIsInstance(
        plan.dp_event.event, dp_accounting.PoissonSampledDpEvent
    )
    self.assertAlmostEqual(plan.dp_event.event.sampling_probability, 0.4)
    spent_epsilon = (
        dp_accounting.rdp.RdpAccountant()
        .compose(probe_event)
        .compose(plan.dp_event)
        .get_epsilon(delta)
    )
    self.assertLessEqual(spent_epsilon, epsilon + 1e-6)
    self.assertGreater(spent_epsilon, epsilon - 0.01)

  @parameterized.named_parameters(
      ("empty_dataset", {"train_size": 0}),
      ("negative_dataset", {"train_size": -1}),
      ("zero_batch", {"batch_size": 0}),
      ("negative_batch", {"batch_size": -1}),
      ("zero_accumulation", {"gradient_accumulation_steps": 0}),
      ("negative_accumulation", {"gradient_accumulation_steps": -1}),
      ("zero_epochs", {"epochs": 0}),
      ("negative_epochs", {"epochs": -1}),
      ("effective_batch_exceeds_dataset", {"batch_size": 6}),
  )
  def test_calibration_rejects_invalid_training_sizes(self, overrides):
    kwargs = dict(
        train_size=10,
        batch_size=2,
        gradient_accumulation_steps=2,
        epochs=2,
        clipping_norm=1.0,
        probe_dp_event=dp_accounting.NoOpDpEvent(),
        epsilon=3.0,
        delta=1e-5,
    )
    kwargs.update(overrides)
    with self.assertRaises(ValueError):
      dpsapf_validate._calibrate_training_config(**kwargs)

  @parameterized.parameters(0.5, 1.0)
  def test_calibration_rejects_exhausted_probe_budget(self, budget_fraction):
    probe_event = accounting.dpsgd_event(
        noise_multiplier=1.0, iterations=1, sampling_prob=0.5
    )
    delta = 1e-5
    probe_epsilon = (
        dp_accounting.rdp.RdpAccountant()
        .compose(probe_event)
        .get_epsilon(delta)
    )
    with mock.patch.object(
        dp_accounting, "calibrate_dp_mechanism"
    ) as calibrate:
      with self.assertRaises(ValueError):
        dpsapf_validate._calibrate_training_config(
            train_size=10,
            batch_size=2,
            gradient_accumulation_steps=2,
            epochs=2,
            clipping_norm=1.0,
            probe_dp_event=probe_event,
            epsilon=probe_epsilon * budget_fraction,
            delta=delta,
        )
    calibrate.assert_not_called()

  @parameterized.named_parameters(
      ("mixed_empty", [0, 1, 7, 2, 0], 3, 9),
      ("all_empty", [0, 0], 3, 3),
  )
  def test_padding_covers_batches_and_handles_empty_draws(
      self, sizes, microbatch_size, expected_multiple
  ):
    sampler = mock.Mock(spec=["batch_iterator"])
    sampler.batch_iterator.return_value = (np.arange(size) for size in sizes)
    multiple = dpsapf_validate._training_padding_multiple(
        sampler,
        10,
        microbatch_size=microbatch_size,
        sampling_seed=np.random.SeedSequence(42),
    )
    self.assertEqual(multiple, expected_multiple)

  @parameterized.parameters(1, 3)
  def test_fit_preserves_samples_and_updates_with_one_padded_shape(
      self, microbatch_size
  ):
    inputs = keras.Input(shape=(3,))
    dense = keras.layers.Dense(2)
    model = keras.Model(inputs, dense(inputs))
    dense.kernel.assign(
        np.array([[0.5, -0.2], [-0.1, 0.3], [0.2, 0.4]], np.float32)
    )
    dense.enable_lora(rank=1)
    dense.bias.trainable = False
    dense.lora_kernel_a.assign(np.full((3, 1), 0.3, np.float32))
    dense.lora_kernel_b.assign(np.zeros((1, 2), np.float32))
    dataset = {
        "x": np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 0.0],
                [2.0, 0.0, 1.0],
            ],
            np.float32,
        ),
        "y": np.array([0, 1, 1, 0, 0, 1, 1, 0], np.int32),
        "sw": np.ones(8, np.float32),
    }
    dataset = jax.tree.map(lambda x: np.concatenate([x] * 8), dataset)
    config = training.BandMFConfig(
        strategy=[1.0],
        iterations=12,
        expected_participations=6.0,
        noise_multiplier=0.1,
        l2_clip_norm=1.0,
        normalize_by=32,
        rescale_to_unit_norm=False,
    )
    initial_trainable = [
        np.array(variable.value) for variable in model.trainable_variables
    ]
    initial_frozen = [
        np.array(variable.value) for variable in model.non_trainable_variables
    ]
    real_fit = training.DPTrainer.fit
    with mock.patch.object(
        training.DPTrainer, "fit", autospec=True, side_effect=real_fit
    ) as fit, mock.patch.object(
        jax, "block_until_ready", wraps=jax.block_until_ready
    ) as synchronize, mock.patch.object(
        training, "_get_batch", wraps=training._get_batch
    ) as get_batch, mock.patch(
        "builtins.print"
    ) as print_progress:
      state = dpsapf_validate._fit_keras_model(
          model,
          dataset,
          config,
          microbatch_size=microbatch_size,
          learning_rate=0.05,
          seed=42,
      )

    self.assertEqual(synchronize.call_count, config.iterations)
    print_progress.assert_has_calls([
        mock.call("DPTrainer update 1/12 completed.", flush=True),
        mock.call("DPTrainer update 10/12 completed.", flush=True),
        mock.call("DPTrainer update 12/12 completed.", flush=True),
    ])
    self.assertEqual(print_progress.call_count, 4)
    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), config.iterations)
    self.assertLen(initial_trainable, 2)
    self.assertLen(state.params, len(initial_trainable))
    for initial, variable, param in zip(
        initial_trainable, model.trainable_variables, state.params, strict=True
    ):
      self.assertFalse(np.array_equal(param, initial))
      np.testing.assert_array_equal(variable, param)
    for initial, variable in zip(
        initial_frozen, model.non_trainable_variables, strict=True
    ):
      np.testing.assert_array_equal(variable, initial)

    predictions = model(dataset["x"], training=False)
    stateless_predictions, _ = model.stateless_call(
        state.params,
        tuple(variable.value for variable in model.non_trainable_variables),
        dataset["x"],
        training=False,
    )
    self.assertTrue(np.all(np.isfinite(predictions)))
    np.testing.assert_allclose(predictions, stateless_predictions, rtol=1e-6)
    fit.assert_called_once()
    self.assertFalse(fit.call_args.kwargs["precompile"])
    trainer = fit.call_args.args[0]
    self.assertEqual(trainer.performance_flags.microbatch_size, microbatch_size)
    self.assertIsInstance(trainer.compilation_strategy, training.PadToMultiple)
    rng = np.random.default_rng(fit.call_args.kwargs["rng_or_seed"])
    rng.integers(2**63)  # DPTrainer's loss PRNG seed draw.
    expected_batches = list(
        trainer.plan.batch_selection_strategy.batch_iterator(64, rng=rng)
    )
    # This fixture straddles the old 32-example compilation boundary.
    old_shapes = {32 * math.ceil(len(batch) / 32) for batch in expected_batches}
    self.assertGreater(len(old_shapes), 1)
    actual_batches = [call.args[1] for call in get_batch.call_args_list]
    self.assertLen(actual_batches, config.iterations)
    self.assertLen({len(batch) for batch in actual_batches}, 1)
    for expected, actual in zip(expected_batches, actual_batches, strict=True):
      self.assertEqual(len(actual) % microbatch_size, 0)
      np.testing.assert_array_equal(
          np.sort(actual[actual >= 0]), np.sort(expected)
      )

    # Padding must not change deterministic loss/gradient/noise semantics.
    reference_trainer = dataclasses.replace(
        trainer,
        compilation_strategy=training.PadToMultiple(
            multiple=math.lcm(32, microbatch_size)
        ),
    )
    reference_state = reference_trainer.fit(
        dataset,
        initial_trainable,
        rng_or_seed=fit.call_args.kwargs["rng_or_seed"],
        precompile=False,
    )
    for actual, expected in zip(
        jax.tree.leaves(state), jax.tree.leaves(reference_state), strict=True
    ):
      if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
        actual = jax.random.key_data(actual)
        expected = jax.random.key_data(expected)
      if np.issubdtype(actual.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
      else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    self.assertIsNotNone(trainer.performance_flags.noise_seed)
    self.assertIsNotNone(fit.call_args.kwargs["rng_or_seed"])
    self.assertNotEqual(
        np.random.default_rng(trainer.performance_flags.noise_seed).integers(
            2**32
        ),
        np.random.default_rng(fit.call_args.kwargs["rng_or_seed"]).integers(
            2**32
        ),
    )


if __name__ == "__main__":
  absltest.main()
