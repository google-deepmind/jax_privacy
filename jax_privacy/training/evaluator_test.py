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

from collections.abc import  Mapping

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import typing
from jax_privacy.training import devices
from jax_privacy.training import dp_updater
from jax_privacy.training import evaluator as evaluator_py
from jax_privacy.training import experiment_config
from jax_privacy.training import forward
from jax_privacy.training import metrics as metrics_module
from jax_privacy.training import optimizer_config
import optax


INPUT_SIZE = 3
LOCAL_BATCH_SIZE = 2
NUM_CLASSES = 7
NUM_DEVICES = 4
NUM_TEST_SAMPLES = 50


def model_fn(inputs, is_training=False):
  del is_training  # unused
  return hk.nets.MLP(output_sizes=[INPUT_SIZE, 2, NUM_CLASSES])(inputs)


@chex.dataclass(frozen=True)
class _TestData:
  """Artificial supervised learning data batch for testing.

  Attributes:
    features: Batch of network inputs.
    label: Batch of labels associated with the inputs.
  """

  features: chex.Array
  label: chex.Array


class _ForwardFnWithRng(forward.ForwardFn):
  """Simple forward function for testing."""

  def __init__(self, net: hk.TransformedWithState):
    self._net = net

  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: _TestData,
  ) -> tuple[hk.Params, hk.State]:
    return self._net.init(rng_key, inputs.features, is_training=True)

  def train_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng_per_example: chex.PRNGKey,
      inputs: _TestData,
  ) -> tuple[typing.Loss, tuple[hk.State, typing.Metrics]]:
    logits, network_state = self._net.apply(
        params, network_state, rng_per_example, inputs.features,
        is_training=True)
    loss = optax.softmax_cross_entropy(logits, inputs.label)

    metrics = typing.Metrics(
        scalars_avg=self._metrics(logits, inputs.label),
        per_example={'loss': loss},
    )
    return jnp.mean(loss), (network_state, metrics)

  def eval_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng: chex.PRNGKey,
      inputs: _TestData,
  ) -> typing.Metrics:
    # Adapt the forward function to echo the per-example random key.
    logits, unused_network_state = self._net.apply(
        params, network_state, rng, inputs.features)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, inputs.label))

    return typing.Metrics(
        per_example={'logits': logits},
        scalars_avg={
            'loss': loss,
            'rng': rng,  # Echo per-example random key, for testing.
            **self._metrics(logits, inputs.label),
        },
    )

  def _metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    """Evaluates topk accuracy."""
    # NB: labels are one-hot encoded.
    acc1, acc5 = metrics_module.topk_accuracy(logits, labels, topk=(1, 5))
    return {'acc1': acc1, 'acc5': acc5}


def _test_data(num_batches, local_batch_size, seed=9273):
  """Generates dummy data for testing purposes."""
  prng_seq = hk.PRNGSequence(seed)
  batches = []
  for _ in range(num_batches):
    rng = next(prng_seq)
    features = jax.random.normal(
        rng,
        [NUM_DEVICES, local_batch_size, INPUT_SIZE],
    )
    rng = next(prng_seq)
    label = jax.random.randint(
        rng,
        [NUM_DEVICES, local_batch_size],
        minval=0,
        maxval=NUM_CLASSES,
    )
    label = jax.nn.one_hot(label, NUM_CLASSES)
    batches.append(_TestData(features=features, label=label))
  return batches


class _MockEvaluator(evaluator_py.AbstractEvaluator):
  """Defined to prevent error."""

  def evaluate_dataset(self, updater_state, ds_iterator):
    return self._evaluate_batch(updater_state, next(ds_iterator))


class EvaluatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(NUM_DEVICES)
    rng = jax.random.PRNGKey(84452)
    self.rng, self.rng_init = jax.random.split(rng, 2)

    net = hk.transform_with_state(model_fn)
    self.forward_fn = _ForwardFnWithRng(net)

  def test_evaluation(self):
    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)[0]
    dp_config = experiment_config.DpConfig.deactivated()
    device_layout = devices.DeviceLayout()
    grad_computer = gradients.DpsgdGradientComputer(
        clipping_norm=dp_config.clipping_norm,
        noise_multiplier=dp_config.algorithm.noise_multiplier,
        rescale_to_unit_norm=dp_config.rescale_to_unit_norm,
        per_example_grad_method=dp_config.per_example_grad_method,
    )
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=LOCAL_BATCH_SIZE * NUM_DEVICES,
            per_device_per_step=LOCAL_BATCH_SIZE,
        ),
        rng=jax.random.PRNGKey(42),
        weight_decay=0.0,
        optimizer_config=optimizer_config.sgd_config(
            lr=optimizer_config.constant_lr_config(1.0),
        ),
        logging_config=experiment_config.LoggingConfig(),
        max_num_updates=10,
        num_training_samples=32,
        grad_computer=grad_computer,
        device_layout=device_layout
    )
    updater_state, _ = updater.init(rng=self.rng_init, inputs=inputs)

    evaluator = _MockEvaluator(
        forward_fn=self.forward_fn,
        rng=jax.random.PRNGKey(42),
    )

    inputs = _test_data(num_batches=2, local_batch_size=LOCAL_BATCH_SIZE)
    metrics = evaluator.evaluate_dataset(
        updater_state=updater_state, ds_iterator=iter(inputs)
    )

    # The different devices' outputs should arise from different random keys.
    for j in range(1, NUM_DEVICES):
      self.assertNotAlmostEqual(
          metrics['last'].scalars_avg['rng'][0],
          metrics['last'].scalars_avg['rng'][j])

if __name__ == '__main__':
  absltest.main()
