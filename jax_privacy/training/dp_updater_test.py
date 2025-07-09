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

from collections.abc import Iterable, Mapping
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import grad_clipping
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import typing
from jax_privacy.training import averaging
from jax_privacy.training import devices
from jax_privacy.training import dp_updater
from jax_privacy.training import experiment_config
from jax_privacy.training import forward
from jax_privacy.training import metrics as metrics_module
from jax_privacy.training import optimizer_config
from jaxline import utils
import more_itertools as itertools
import numpy as np
import optax
import scipy.stats as spst


INPUT_SIZE = 3
LOCAL_BATCH_SIZE = 2
NUM_CLASSES = 7
NUM_DEVICES = 4
NUM_TEST_SAMPLES = 50
RUN_STATISTICAL_TESTS = False


def _standard_updater_kwargs(
    *,
    learning_rate: float = 1.0,
    weight_decay: float = 0.0,
):
  return {
      'weight_decay': weight_decay,
      'optimizer_config': optimizer_config.sgd_config(
          lr=optimizer_config.constant_lr_config(learning_rate),
      ),
      'logging_config': experiment_config.LoggingConfig(),
      'max_num_updates': 10,
      'num_training_samples': 128,
  }


def _flatten_tree(tree):
  return jnp.concatenate(
      [jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)]
  )


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
        params,
        network_state,
        rng_per_example,
        inputs.features,
        is_training=True,
    )
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
        params, network_state, rng, inputs.features
    )
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


def assert_close_calibrated(value, ref, expected, rtol=2, atol=1e-5):
  """Check that |value - expected| <= rtol * max(|ref - expected|) + atol."""
  delta = jnp.abs(value - expected)
  max_delta_allowed = rtol * jnp.max(jnp.abs(ref - expected)) + atol
  np.testing.assert_array_less(delta, max_delta_allowed)


def assert_trees_all_close(tree_1, tree_2, rtol=2e-2, atol=1e-5):
  """Check closeness up to *both* absolute and relative tolerance values."""
  chex.assert_trees_all_close(tree_1, tree_2, atol=atol, rtol=0)
  chex.assert_trees_all_close(tree_1, tree_2, rtol=rtol, atol=0)


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


def _make_gradient_computer(
    *,
    noise_multiplier: float | None,
    clipping_norm: float | None = 0.1,
    rescale_to_unit_norm: bool = False,
) -> gradients.GradientComputer:
  return gradients.DpsgdGradientComputer(
      clipping_norm=clipping_norm,
      noise_multiplier=noise_multiplier,
      rescale_to_unit_norm=rescale_to_unit_norm,
      per_example_grad_method=grad_clipping.VECTORIZED,
  )


class UpdaterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(NUM_DEVICES)
    rng = jax.random.PRNGKey(84452)
    self.rng, self.rng_init = jax.random.split(rng, 2)

    self.net = hk.transform_with_state(model_fn)
    self.forward_fn = _ForwardFnWithRng(self.net)

  def init_with_updater(self, updater: dp_updater.Updater):
    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)[0]
    self._updater_state, self._step_on_host = updater.init(
        rng=self.rng_init, inputs=inputs
    )

  def run_updater(
      self,
      updater: dp_updater.Updater,
      data: Iterable[_TestData],
      num_steps: int = 1,
      return_all_params: bool = False,
  ):
    """Runs the updater on the data given in argument."""

    state = self._updater_state
    step_on_host = self._step_on_host
    all_params = [utils.get_first(state.params)]
    data_iterator = iter(data)
    for _ in range(num_steps):
      # Args are donated. Take copies so that we can reuse them.
      state, unused_scalars, step_on_host = updater.update(
          state=jax.tree_util.tree_map(jnp.copy, state),
          inputs_producer=functools.partial(next, data_iterator),
          step_on_host=step_on_host,
      )
      all_params.append(utils.get_first(state.params))

    return all_params if return_all_params else all_params[-1]

  @parameterized.named_parameters(
      ('no_accumulation_no_weight_decay', 1, 0.0),
      ('no_accumulation_with_weight_decay', 1, 1000.0),
      ('with_accumulation_no_weight_decay', 3, 0.0),
      ('with_accumulation_with_weight_decay', 3, 1000.0),
  )
  def test_accumulation(self, num_accumulations, weight_decay):
    batch_size = LOCAL_BATCH_SIZE * NUM_DEVICES * num_accumulations

    # When using weight-decay, ensure that it is the dominant term in the update
    # by using a small learning-rate (downweighs the importance of gradients),
    # so that we can detect it.
    learning_rate = 0.1 if not weight_decay else 0.1 / weight_decay
    device_layout = devices.DeviceLayout()
    updater_no_noise = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=batch_size,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            clipping_norm=1.0,
            noise_multiplier=0.0,
        ),
        rng=jax.random.PRNGKey(42),
        device_layout=device_layout,
        **_standard_updater_kwargs(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
    )

    self.init_with_updater(updater_no_noise)

    data = _test_data(
        num_batches=num_accumulations,
        local_batch_size=LOCAL_BATCH_SIZE,
    )

    output_params = self.run_updater(
        updater_no_noise, data, return_all_params=True
    )

    # Check that parameters are unchanged during accumulation steps.
    for params in output_params[1:-1]:
      chex.assert_trees_all_equal(params, output_params[0])

    # Check that parameters have changed during the update step.
    with self.assertRaises(AssertionError):
      chex.assert_trees_all_close(
          output_params[-1],
          output_params[0],
          rtol=0.1,
      )

  @parameterized.named_parameters(
      ('no_accumulation', 1),
      ('with_accumulation', 5),
  )
  def test_noise(self, num_accumulations):
    if not RUN_STATISTICAL_TESTS:
      self.skipTest('Skipping the slow statistical tests.')
    std = 0.3
    clipping_norm = 0.1
    batch_size = LOCAL_BATCH_SIZE * NUM_DEVICES * num_accumulations

    device_layout = devices.DeviceLayout()
    updater_no_noise = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=batch_size,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            clipping_norm=clipping_norm,
            noise_multiplier=0.0,
        ),
        rng=jax.random.PRNGKey(42),
        device_layout=device_layout,
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater_no_noise)

    data = _test_data(
        num_batches=num_accumulations,
        local_batch_size=LOCAL_BATCH_SIZE,
    )

    # Run one pass of the updater over the data with no noise.
    params_no_noise = self.run_updater(updater_no_noise, data)

    # Multiple realizations for different rngs.
    noise_samples = []
    device_layout = devices.DeviceLayout()
    for i in range(NUM_TEST_SAMPLES):
      updater_noise = dp_updater.Updater(
          forward_fn=self.forward_fn,
          batch_size_config=experiment_config.BatchSizeTrainConfig(
              total=batch_size,
              per_device_per_step=LOCAL_BATCH_SIZE,
              scale_schedule=None,
          ),
          grad_computer=_make_gradient_computer(
              clipping_norm=clipping_norm,
              noise_multiplier=std,
          ),
          rng=jax.random.split(jax.random.PRNGKey(i))[0],
          device_layout=device_layout,
          **_standard_updater_kwargs(),
      )
      # Run one pass of the updater over data with noise using rng_iteration.
      params_noise = self.run_updater(updater_noise, data)
      # The difference with params_no_noise should only contain the noise.
      noise_samples.append(
          _flatten_tree(params_noise) - _flatten_tree(params_no_noise)
      )

    noise_samples = jnp.stack(noise_samples)

    std_expected = std * clipping_norm / batch_size

    # Use synthetic noise as a reference to calibrate the precision required
    # to pass the test.
    synthetic_noise = std_expected * jax.random.normal(
        self.rng, noise_samples.shape
    )

    # Sanity check: synthetic noise passes KS goodness-of-fit test.
    _, p_synthetic = spst.kstest(
        jnp.ravel(synthetic_noise) / std_expected, 'norm'
    )
    self.assertGreater(p_synthetic, 0.05)

    # Run KS goodness-of-fit test on noise introduced by DP-SGD.
    _, p_dpsgd = spst.kstest(jnp.ravel(noise_samples) / std_expected, 'norm')
    # Reject null hypothesis "implementation is correct" if p-value <= 0.05.
    self.assertGreater(p_dpsgd, 0.05)

    # Statistics per coordinate, across samples (rng instances)
    # (to test that the noise is independent across rng instances).
    mean_per_coordinate = jnp.mean(noise_samples, axis=0)
    std_per_coordinate = jnp.std(noise_samples, axis=0)
    mean_per_coordinate_ref = jnp.mean(synthetic_noise, axis=0)
    std_per_coordinate_ref = jnp.std(synthetic_noise, axis=0)

    # Statistics per sample (rng instance), across coordinates
    # (to test that the noise is independent across coordinates).
    mean_per_sample = jnp.mean(noise_samples, axis=1)
    std_per_sample = jnp.std(noise_samples, axis=1)
    mean_per_sample_ref = jnp.mean(synthetic_noise, axis=1)
    std_per_sample_ref = jnp.std(synthetic_noise, axis=1)

    # Statistics across both samples and coordinates.
    total_mean = jnp.mean(noise_samples)
    total_mean_ref = jnp.mean(synthetic_noise)
    total_std = jnp.std(noise_samples)
    total_std_ref = jnp.std(synthetic_noise)

    assert_close_calibrated(
        value=mean_per_coordinate,
        ref=mean_per_coordinate_ref,
        expected=0.0,
    )

    assert_close_calibrated(
        value=std_per_coordinate,
        ref=std_per_coordinate_ref,
        expected=std_expected,
    )

    assert_close_calibrated(
        value=mean_per_sample,
        ref=mean_per_sample_ref,
        expected=0.0,
    )

    assert_close_calibrated(
        value=std_per_sample,
        ref=std_per_sample_ref,
        expected=std_expected,
    )

    assert_close_calibrated(
        value=total_mean,
        ref=total_mean_ref,
        expected=0.0,
    )

    assert_close_calibrated(
        value=total_std,
        ref=total_std_ref,
        expected=std_expected,
    )

  @parameterized.parameters(0.01, 0.1, 1.0, 10.0)
  def test_clipping(self, clipping_norm):
    device_layout = devices.DeviceLayout()
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=LOCAL_BATCH_SIZE * NUM_DEVICES,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            clipping_norm=clipping_norm,
            noise_multiplier=0.0,
        ),
        rng=jax.random.PRNGKey(42),
        device_layout=device_layout,
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params = self.run_updater(updater, data)
    initial_params = utils.get_first(self._updater_state.params)
    # Invert SGD equation (with lr=1) to find gradients.
    grads_updater = _flatten_tree(initial_params) - _flatten_tree(params)

    # Only one mini-batch in this test.
    inputs = data[0]
    # Merge two leading dimensions to put all data on a single device.
    inputs_single_device = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
        inputs,
    )

    def forward_per_sample(p):
      logits, unused_network_state = self.net.apply(
          p,
          self._updater_state.network_state,
          self.rng,
          inputs_single_device.features,
          is_training=True,
      )

      loss_per_sample = optax.softmax_cross_entropy(
          logits, inputs_single_device.label
      )

      # Check that the batch dimension is correct.
      chex.assert_shape(loss_per_sample, [LOCAL_BATCH_SIZE * NUM_DEVICES])

      return loss_per_sample

    def clip_global_norm(tree):
      l2_norm = optax.global_norm(tree)
      coeff = jnp.minimum(clipping_norm / l2_norm, 1.0)
      return jax.tree_util.tree_map(lambda x: x * coeff, tree)

    # Compute Jacobian of the loss function.
    jacobian = jax.jacobian(forward_per_sample)(initial_params)
    # Clip Jacobian per sample.
    jacobian_clipped = jax.vmap(clip_global_norm)(jacobian)
    # Average over samples.
    grads_manual = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0),
        jacobian_clipped,
    )
    # Flatten to compare with grads_updater.
    grads_manual = _flatten_tree(grads_manual)

    assert_trees_all_close(grads_updater, grads_manual)

  def test_frozen_params(self):
    train_only_layer = 'mlp/~/linear_1'
    device_layout = devices.DeviceLayout()
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=LOCAL_BATCH_SIZE * NUM_DEVICES,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            noise_multiplier=0.1,
        ),
        rng=jax.random.PRNGKey(42),
        is_trainable=(
            lambda module_name, *args: module_name == train_only_layer
        ),
        device_layout=device_layout,
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)
    params = self.run_updater(updater, data)

    initial_params = utils.get_first(self._updater_state.params)

    count_trainable, count_frozen = 0, 0
    for layer_name in params:
      params_layer = params[layer_name]
      initial_params_layer = initial_params[layer_name]
      if layer_name != train_only_layer:
        # This layer should be frozen.
        count_frozen += 1
        assert_trees_all_close(params_layer, initial_params_layer)
      else:
        # This layer should be updated.
        count_trainable += 1
        chex.assert_trees_all_equal_comparator(
            lambda x1, x2: jnp.linalg.norm(x1 - x2) > 1e-4,
            lambda x1, x2: (
                'Parameters should have been updated, but they were not, they'
                ' are too close, norm(x1 - x2) <= 1e-4:'
                f'\n{x1=}\n{x2=}\n{jnp.linalg.norm(x1 - x2)=}'
            ),
            params_layer,
            initial_params_layer,
        )
    self.assertEqual(count_trainable, 1)
    self.assertEqual(count_frozen, 2)

  @parameterized.parameters(0.01, 0.1, 1.0, 10.0)
  # TODO: explore why 0.01 and 0.1 clipping norms require higher rtol
  def test_rescaling(self, clipping_norm):
    noise_std = 0.1

    def make_updater(*, rescale_to_unit_norm: bool):
      device_layout = devices.DeviceLayout()
      return dp_updater.Updater(
          forward_fn=self.forward_fn,
          batch_size_config=experiment_config.BatchSizeTrainConfig(
              total=LOCAL_BATCH_SIZE * NUM_DEVICES,
              per_device_per_step=LOCAL_BATCH_SIZE,
              scale_schedule=None,
          ),
          grad_computer=_make_gradient_computer(
              clipping_norm=clipping_norm,
              noise_multiplier=noise_std,
              rescale_to_unit_norm=rescale_to_unit_norm,
          ),
          rng=jax.random.PRNGKey(42),
          device_layout=device_layout,
          **_standard_updater_kwargs(),
      )

    updater_no_rescaling = make_updater(rescale_to_unit_norm=False)
    updater_with_rescaling = make_updater(rescale_to_unit_norm=True)

    self.init_with_updater(updater_no_rescaling)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params_with_rescaling = self.run_updater(updater_with_rescaling, data)
    params_no_rescaling = self.run_updater(updater_no_rescaling, data)
    initial_params = utils.get_first(self._updater_state.params)

    # Invert SGD equation (with lr=1) to find gradients.
    grads_with_rescaling = _flatten_tree(initial_params) - _flatten_tree(
        params_with_rescaling
    )
    grads_no_rescaling = _flatten_tree(initial_params) - _flatten_tree(
        params_no_rescaling
    )
    grads_manual_rescaling = grads_no_rescaling / clipping_norm

    assert_trees_all_close(grads_with_rescaling, grads_manual_rescaling)

  def test_evaluation(self):
    forward_fn = _ForwardFnWithRng(self.net)
    device_layout = devices.DeviceLayout()
    updater = dp_updater.Updater(
        forward_fn=forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=LOCAL_BATCH_SIZE * NUM_DEVICES,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            clipping_norm=None,
            noise_multiplier=None,
        ),
        rng=jax.random.PRNGKey(42),
        device_layout=device_layout,
        **_standard_updater_kwargs(),
    )
    self.init_with_updater(updater)

  def test_average_init_takes_copy(self):
    device_layout = devices.DeviceLayout()
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batch_size_config=experiment_config.BatchSizeTrainConfig(
            total=LOCAL_BATCH_SIZE * NUM_DEVICES,
            per_device_per_step=LOCAL_BATCH_SIZE,
            scale_schedule=None,
        ),
        grad_computer=_make_gradient_computer(
            clipping_norm=None,
            noise_multiplier=None,
        ),
        rng=jax.random.PRNGKey(42),
        device_layout=device_layout,
        **_standard_updater_kwargs(),
        averaging_configs={
            'polyak': averaging.PolyakAveragingConfig(),
        },
    )
    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)
    state, unused_step_on_host = updater.init(self.rng_init, inputs[0])
    params = state.params
    chex.assert_trees_all_close(params, state.params_avg['polyak'])

    # Assert that the average survives even when the original is donated.
    jax.tree_util.tree_map(lambda x: x.delete(), params)
    jax.device_get(state.params_avg['polyak'])

  def test_state_with_accumulation(self):
    update_every = 3
    batch_size = update_every * LOCAL_BATCH_SIZE * NUM_DEVICES

    def make_updater(local_batch_size: int) -> dp_updater.Updater:
      device_layout = devices.DeviceLayout()
      return dp_updater.Updater(
          forward_fn=self.forward_fn,
          batch_size_config=experiment_config.BatchSizeTrainConfig(
              total=batch_size,
              per_device_per_step=local_batch_size,
              scale_schedule=None,
          ),
          grad_computer=_make_gradient_computer(
              clipping_norm=None,
              noise_multiplier=None,
          ),
          rng=jax.random.PRNGKey(42),
          device_layout=device_layout,
          **_standard_updater_kwargs(),
          averaging_configs={
              'polyak': averaging.PolyakAveragingConfig(start_step=2),
              'ema': averaging.ExponentialMovingAveragingConfig(
                  decay=0.9,
                  start_step=2,
              ),
          },
      )

    updater_with_accumulation = make_updater(LOCAL_BATCH_SIZE)
    updater_without_accumulation = make_updater(batch_size // NUM_DEVICES)

    inputs = _test_data(num_batches=7, local_batch_size=LOCAL_BATCH_SIZE)
    state, step_on_host = updater_with_accumulation.init(
        rng=self.rng_init, inputs=inputs[0]
    )

    inputs_iterator = iter(inputs[1:])
    state_1, metrics_1, step_on_host_1 = updater_with_accumulation.update(
        state=state,
        inputs_producer=functools.partial(next, inputs_iterator),
        step_on_host=step_on_host,
    )

    state, step_on_host = updater_without_accumulation.init(
        rng=self.rng_init, inputs=inputs[0]
    )
    inputs_reshaped = []
    for list_of_batches in itertools.batched(inputs[1:], update_every):
      inputs_reshaped.append(
          jax.tree_util.tree_map(
              lambda *x: jnp.concatenate(x, axis=1), *list_of_batches
          )
      )
    inputs_iterator = iter(inputs_reshaped)
    state_2, metrics_2, step_on_host_2 = updater_without_accumulation.update(
        state=state,
        inputs_producer=functools.partial(next, inputs_iterator),
        step_on_host=step_on_host,
    )

    exclude_scalars = ('update_every',)  # not equal
    scalars_1 = {
        k: v for k, v in metrics_1.scalars.items() if k not in exclude_scalars
    }
    scalars_2 = {
        k: v for k, v in metrics_2.scalars.items() if k not in exclude_scalars
    }
    chex.assert_trees_all_close_ulp(state_1, state_2, maxulp=32)
    chex.assert_trees_all_close_ulp(scalars_1, scalars_2, maxulp=32)
    chex.assert_trees_all_equal(step_on_host_1, step_on_host_2)
    np.testing.assert_array_equal(
        metrics_1.scalars_last['update_every'], update_every
    )
    np.testing.assert_array_equal(metrics_2.scalars_last['update_every'], 1)


if __name__ == '__main__':
  absltest.main()
