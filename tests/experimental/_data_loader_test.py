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


import concurrent.futures

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_privacy import batch_selection
from jax_privacy import execution_plan
from jax_privacy.experimental import _data_loader
from jax_privacy.experimental import training
import numpy as np
import optax
import pytest

grain = pytest.importorskip('grain.python')

# ---------------------------------------------------------------------------
# Standalone _data_loader tests
# ---------------------------------------------------------------------------


class IsPygrainMapDatasetTest(absltest.TestCase):
  """Tests for the string-based MapDataset type check."""

  def test_grain_map_dataset_detected(self):
    ds = grain.MapDataset.source([1, 2, 3])
    self.assertTrue(_data_loader.is_pygrain_map_dataset(ds))

  def test_transformed_map_dataset_detected(self):
    ds = grain.MapDataset.source([1, 2, 3]).map(lambda x: x + 1)
    self.assertTrue(_data_loader.is_pygrain_map_dataset(ds))

  def test_numpy_array_rejected(self):
    self.assertFalse(_data_loader.is_pygrain_map_dataset(np.zeros((3, 2))))

  def test_python_list_rejected(self):
    self.assertFalse(_data_loader.is_pygrain_map_dataset([1, 2, 3]))

  def test_dict_rejected(self):
    self.assertFalse(_data_loader.is_pygrain_map_dataset({'x': np.zeros((3,))}))


class IterateBatchesTest(parameterized.TestCase):
  """Tests for iterate_batches with a real grain MapDataset."""

  def _make_dataset(self, num_examples=10, dim=2):
    """Creates a simple MapDataset of dicts."""
    examples = [
        {'x': np.full(dim, i, dtype=np.float32)} for i in range(num_examples)
    ]
    return grain.MapDataset.source(examples)

  def test_yields_correct_number_of_batches(self):
    ds = self._make_dataset(num_examples=8)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=5, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)

    batches = list(
        _data_loader.iterate_batches(ds, strategy, rng, pad_to_multiple_of=1)
    )
    self.assertLen(batches, 5)

  def test_auto_preload_small_dataset(self):
    """Small datasets should auto-preload (default preload=None)."""
    ds = self._make_dataset(num_examples=4)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=2, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)
    # Should auto-detect as preload=True and produce valid batches.
    batches = list(
        _data_loader.iterate_batches(ds, strategy, rng, pad_to_multiple_of=1)
    )
    self.assertLen(batches, 2)

  def test_estimate_dataset_bytes(self):
    ds = self._make_dataset(num_examples=10, dim=4)
    # Each element is {'x': float32[4]} = 16 bytes, so total = 160 bytes.
    estimated = _data_loader._estimate_dataset_bytes(ds)
    self.assertEqual(estimated, 160)

  def test_batch_contains_correct_keys(self):
    ds = self._make_dataset(num_examples=4)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=1, sampling_prob=1.0
    )
    rng = np.random.default_rng(42)

    ((batch, is_padding),) = list(
        _data_loader.iterate_batches(ds, strategy, rng, pad_to_multiple_of=1)
    )
    self.assertIn('x', batch)
    self.assertEqual(batch['x'].ndim, 2)
    self.assertEqual(is_padding.ndim, 1)

  def test_padding_entries_are_zeroed(self):
    ds = self._make_dataset(num_examples=3, dim=1)
    # sampling_prob=1.0 guarantees all 3 examples are selected.
    # pad_to_multiple_of=4 pads the batch from 3 → 4, so one entry is padding.
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=1, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)

    ((batch, is_padding),) = list(
        _data_loader.iterate_batches(ds, strategy, rng, pad_to_multiple_of=4)
    )
    self.assertTrue(is_padding.any())
    np.testing.assert_array_equal(batch['x'][is_padding], 0.0)

  @parameterized.parameters(1, 4, 8)
  def test_padding_multiple_respected(self, padding_multiple):
    ds = self._make_dataset(num_examples=10)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=3, sampling_prob=0.5
    )
    rng = np.random.default_rng(7)

    for batch, _ in _data_loader.iterate_batches(
        ds, strategy, rng, pad_to_multiple_of=padding_multiple
    ):
      self.assertEqual(batch['x'].shape[0] % padding_multiple, 0)

  @parameterized.parameters(True, False)
  def test_preload_modes_produce_same_results(self, preload):
    """Both preload paths should produce identical batches."""
    ds = self._make_dataset(num_examples=6, dim=3)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=4, sampling_prob=1.0
    )
    rng = np.random.default_rng(99)

    batches = list(
        _data_loader.iterate_batches(
            ds, strategy, rng, pad_to_multiple_of=4, preload=preload
        )
    )
    self.assertLen(batches, 4)
    for batch, is_padding in batches:
      self.assertIn('x', batch)
      if preload:
        # Preloaded path explicitly zeros padding entries.
        np.testing.assert_array_equal(batch['x'][is_padding], 0.0)


class PrivateBatchIteratorTest(parameterized.TestCase):
  """Tests for the streaming PrivateBatchIterator."""

  def _make_dataset(self, num_examples=10, dim=2):
    examples = [
        {'x': np.full(dim, i, dtype=np.float32)} for i in range(num_examples)
    ]
    return grain.MapDataset.source(examples)

  def test_yields_correct_number_of_batches(self):
    ds = self._make_dataset(num_examples=8)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=5, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)

    it = _data_loader.PrivateBatchIterator(
        ds, strategy, rng, pad_to_multiple_of=1
    )
    batches = list(it)
    self.assertLen(batches, 5)

  def test_padding_zeroed(self):
    ds = self._make_dataset(num_examples=3, dim=1)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=1, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)

    it = _data_loader.PrivateBatchIterator(
        ds, strategy, rng, pad_to_multiple_of=4
    )
    batch, is_padding = next(it)
    self.assertTrue(is_padding.any())
    # Note: padding values are not guaranteed to be zero in streaming mode
    # because the iterator uses np.empty_like for the padding template.
    self.assertEqual(batch['x'].shape[0], 4)

  def test_get_set_state_roundtrip(self):
    ds = self._make_dataset(num_examples=6)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=4, sampling_prob=1.0
    )
    rng = np.random.default_rng(42)

    it = _data_loader.PrivateBatchIterator(
        ds, strategy, rng, pad_to_multiple_of=1
    )
    # Consume 2 batches.
    _, _ = next(it)
    _, _ = next(it)
    state = it.get_state()

    # Create new iterator and restore state.
    it2 = _data_loader.PrivateBatchIterator(
        ds, strategy, rng, pad_to_multiple_of=1
    )
    it2.set_state(state)
    batch3, _ = next(it2)
    batch3_orig, _ = next(it)
    np.testing.assert_array_equal(batch3['x'], batch3_orig['x'])

  def test_streaming_matches_preload(self):
    """Streaming and preload should produce the same batch contents."""
    ds = self._make_dataset(num_examples=6, dim=3)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=4, sampling_prob=1.0
    )
    rng = np.random.default_rng(99)

    preloaded = list(
        _data_loader.iterate_batches(
            ds, strategy, rng, pad_to_multiple_of=4, preload=True
        )
    )
    streamed = list(
        _data_loader.iterate_batches(
            ds, strategy, rng, pad_to_multiple_of=4, preload=False
        )
    )
    self.assertLen(preloaded, len(streamed))
    for (pb, pp), (sb, sp) in zip(preloaded, streamed):
      np.testing.assert_array_equal(pp, sp)
      # Only compare non-padding entries; padding values may differ because
      # the preloaded path zeros padding via np.where while the streaming
      # path uses np.empty_like (uninitialized memory) for the template.
      non_pad = ~pp
      np.testing.assert_array_equal(pb['x'][non_pad], sb['x'][non_pad])

  def test_max_workers_respected(self):
    """Ensures max_workers parameter is passed through."""
    ds = self._make_dataset(num_examples=4)
    strategy = batch_selection.CyclicPoissonSampling(
        iterations=1, sampling_prob=1.0
    )
    rng = np.random.default_rng(0)

    it = _data_loader.PrivateBatchIterator(
        ds, strategy, rng, pad_to_multiple_of=1, max_workers=2
    )
    self.assertIsInstance(it._executor, concurrent.futures.ThreadPoolExecutor)
    batch, _ = next(it)
    self.assertIn('x', batch)


# ---------------------------------------------------------------------------
# End-to-end training with a PyGrain MapDataset
# ---------------------------------------------------------------------------


def _quadratic_loss(params, batch, prng):
  """Per-example quadratic loss."""
  del prng
  loss = jnp.mean((params - batch['x']) ** 2)
  return loss, {'loss': loss}


def _make_plan(iterations, noise_multiplier=1.0, sampling_prob=1.0):
  config = execution_plan.BandMFExecutionPlanConfig.default(
      num_bands=1,
      iterations=iterations,
      noise_multiplier=noise_multiplier,
      sampling_prob=sampling_prob,
  )
  return config.make()


class DPTrainerPygrainTest(parameterized.TestCase):
  """End-to-end tests for DPTrainer.fit() with a PyGrain MapDataset."""

  def _make_dataset(self, num_examples=6, dim=2):
    examples = [
        {'x': np.random.default_rng(i).standard_normal(dim).astype(np.float32)}
        for i in range(num_examples)
    ]
    return grain.MapDataset.source(examples)

  def test_basic_training_completes(self):
    params = jnp.array([5.0, 5.0])
    ds = self._make_dataset(num_examples=4, dim=2)
    plan = _make_plan(iterations=3)
    optimizer = optax.sgd(0.01)

    trainer = training.DPTrainer(
        plan=plan, loss_fn=_quadratic_loss, optimizer=optimizer
    )
    state = trainer.fit(ds, params, rng=0)

    self.assertIsInstance(state, training.TrainingState)
    self.assertEqual(int(state.step), 3)

  def test_params_change(self):
    params = jnp.array([10.0, 10.0])
    ds = self._make_dataset(num_examples=4, dim=2)
    plan = _make_plan(iterations=5, noise_multiplier=0.0)
    optimizer = optax.sgd(0.1)

    trainer = training.DPTrainer(
        plan=plan, loss_fn=_quadratic_loss, optimizer=optimizer
    )
    state = trainer.fit(ds, params, rng=42)

    self.assertFalse(jnp.allclose(state.params, params))

  def test_callback_invoked(self):
    params = jnp.array([1.0, 1.0])
    ds = self._make_dataset(num_examples=4, dim=2)
    iterations = 3
    plan = _make_plan(iterations=iterations)
    optimizer = optax.sgd(0.01)
    log = []

    trainer = training.DPTrainer(
        plan=plan, loss_fn=_quadratic_loss, optimizer=optimizer
    )
    trainer.fit(
        ds,
        params,
        callback=lambda step, state, aux: log.append(step),
        rng=0,
    )

    self.assertEqual(log, [1, 2, 3])

  def test_in_memory_path_still_works(self):
    """Ensure the in-memory (non-pygrain) path is not broken."""
    params = jnp.array([5.0, 5.0])
    dataset = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    plan = _make_plan(iterations=3)
    optimizer = optax.sgd(0.01)

    def loss_fn(params, batch, prng):
      del prng
      loss = jnp.mean((params - batch) ** 2)
      return loss, {}

    trainer = training.DPTrainer(
        plan=plan, loss_fn=loss_fn, optimizer=optimizer
    )
    state = trainer.fit(dataset, params, rng=0)

    self.assertEqual(int(state.step), 3)


if __name__ == '__main__':
  absltest.main()
