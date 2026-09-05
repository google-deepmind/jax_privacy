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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dp_accounting
import jax
import jax.numpy as jnp
from jax_privacy import batch_selection
from jax_privacy import saliency


def _sampling(probability):
  return batch_selection.CyclicPoissonSampling(
      sampling_prob=probability,
      iterations=1,
      partition_type=batch_selection.PartitionType.INDEPENDENT,
  )


def _linear_loss(params, batch):
  return sum(param * jnp.sum(batch[..., i]) for i, param in enumerate(params))


def _run_probe(dataset, *, microbatch_size=1, sampling=None, **kwargs):
  num_candidates = dataset.shape[-1]
  if sampling is None:
    sampling = _sampling(1.0)
  defaults = dict(
      loss_fn=_linear_loss,
      dataset=dataset,
      params=tuple(jnp.array(0.0) for _ in range(num_candidates)),
      vote_top_k=1,
      select_top_k=1,
      noise_multiplier=0.0,
      candidate_mask=(True,) * num_candidates,
      prng_key=jax.random.key(0),
      sampling_strategy=sampling,
      microbatch_size=microbatch_size,
  )
  defaults.update(kwargs)
  return saliency.topk_vote_probe(**defaults)


class SaliencyTest(parameterized.TestCase):

  def test_poisson_sample_and_dp_event_share_strategy(self):
    population_size = 8
    probability = 0.5
    sampling = _sampling(probability)
    indices = next(sampling.batch_iterator(population_size, rng=7))
    dataset = jnp.eye(population_size, dtype=jnp.float32)[indices]

    result = _run_probe(
        dataset,
        sampling=sampling,
        select_top_k=2,
        microbatch_size=3,
    )

    scores = dict(result.ranked_scores)
    expected_scores = {i: float(i in indices) for i in range(population_size)}
    self.assertEqual(scores, expected_scores)
    self.assertIsInstance(result.dp_event, dp_accounting.PoissonSampledDpEvent)
    self.assertEqual(result.dp_event.sampling_probability, probability)
    self.assertIsInstance(result.dp_event.event, dp_accounting.GaussianDpEvent)
    self.assertEqual(result.dp_event.event.noise_multiplier, 0.0)

  @parameterized.parameters(None, 1, 2, 4)
  def test_internal_microbatching_matches_unmicrobatched(self, microbatch_size):
    dataset = jnp.array([
        [3.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 0.0],
    ])

    result = _run_probe(
        dataset,
        microbatch_size=microbatch_size,
        select_top_k=2,
    )

    self.assertEqual(result.ranked_scores, [(0, 3.0), (1, 2.0), (2, 1.0)])
    chex.assert_trees_all_equal(result.selected_mask, (True, True, False))

  def test_empty_poisson_sample_is_supported(self):
    result = _run_probe(
        jnp.empty((0, 3), dtype=jnp.float32),
        sampling=_sampling(0.5),
        microbatch_size=4,
    )

    self.assertEqual(dict(result.ranked_scores), {0: 0.0, 1: 0.0, 2: 0.0})

  def test_noise_scale_uses_clipped_grad_sensitivity(self):
    class FakeGradFn:

      def __init__(self):
        self.sensitivity = mock.Mock(return_value=7.0)

      def __call__(self, unused_params, unused_dataset, **unused_kwargs):
        return jnp.zeros((2,), dtype=jnp.float32)

    fake_grad_fn = FakeGradFn()

    class NoNoisePrivatizer:

      def init(self, unused_value):
        return ()

      def update(self, value, state):
        return value, state

    with mock.patch.object(
        saliency.clipping, 'clipped_grad', return_value=fake_grad_fn
    ), mock.patch.object(
        saliency.noise_addition,
        'gaussian_privatizer',
        return_value=NoNoisePrivatizer(),
    ) as privatizer:
      _run_probe(jnp.ones((2, 2)), noise_multiplier=2.0)

    fake_grad_fn.sensitivity.assert_called_once_with(
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
    )
    self.assertEqual(privatizer.call_args.kwargs['stddev'], 14.0)

  def test_argument_validation(self):
    dataset = jnp.ones((2, 3))
    params = (jnp.array(0.0),) * 3
    base = dict(
        loss_fn=_linear_loss,
        dataset=dataset,
        params=params,
        vote_top_k=1,
        select_top_k=1,
        noise_multiplier=1.0,
        candidate_mask=(True, True, True),
        prng_key=jax.random.key(0),
        sampling_strategy=_sampling(1.0),
        microbatch_size=1,
    )
    invalid_cases = (
        ('num_candidates', {'candidate_mask': (False, False, False)}),
        ('vote_top_k', {'vote_top_k': 0}),
        ('select_top_k', {'select_top_k': 4}),
        ('noise_multiplier', {'noise_multiplier': -1.0}),
        ('microbatch_size', {'microbatch_size': 0}),
        ('sampling_probability', {'sampling_strategy': _sampling(0.0)}),
    )
    for message, overrides in invalid_cases:
      with self.subTest(message=message), self.assertRaisesRegex(
          ValueError, message
      ):
        saliency.topk_vote_probe(**(base | overrides))

    with self.assertRaisesRegex(ValueError, 'PyTree structure'):
      saliency.topk_vote_probe(**(base | {'candidate_mask': [True] * 3}))

    with self.assertRaisesRegex(ValueError, 'same size along axis 0'):
      saliency.topk_vote_probe(
          **(base | {'dataset': {'x': dataset, 'y': jnp.ones((3,))}})
      )

  def test_rejects_nonstandard_poisson_strategy(self):
    sampling = batch_selection.CyclicPoissonSampling(
        sampling_prob=0.5,
        iterations=1,
        partition_type=batch_selection.PartitionType.EQUAL_SPLIT,
    )
    with self.assertRaisesRegex(ValueError, 'sampling_partition_type'):
      _run_probe(jnp.ones((2, 2)), sampling=sampling)


if __name__ == '__main__':
  absltest.main()
