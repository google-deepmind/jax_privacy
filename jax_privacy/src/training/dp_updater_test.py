# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

"""Tests the updater."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import forward
from jax_privacy.src.dp_sgd import batching as batching_module
from jax_privacy.src.dp_sgd import gradients
from jax_privacy.src.training import dp_updater
from jax_privacy.src.training import experiment_config
from jax_privacy.src.training import optimizer_config
from jaxline import utils
import numpy as np
import optax
import scipy.stats as spst


INPUT_SIZE = 3
LOCAL_BATCH_SIZE = 2
NUM_CLASSES = 7
NUM_DEVICES = 4
AUGMULT = 5
NUM_TEST_SAMPLES = 18


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
  }


def _flatten_tree(tree):
  return jnp.concatenate(
      [jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def model_fn(inputs, is_training=False):
  del is_training  # unused
  return hk.nets.MLP(output_sizes=[INPUT_SIZE, 10, NUM_CLASSES])(inputs)


# Adapt the forward function to echo the per-example random key.
class _ForwardFnWithRng(forward.MultiClassForwardFn):

  def eval_forward(self, params, network_state, rng, inputs):
    metrics = super().eval_forward(params, network_state, rng, inputs)
    metrics.scalars_avg = {'rng': rng, **metrics.scalars_avg}
    return metrics


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
    images = jax.random.normal(
        rng,
        [NUM_DEVICES, local_batch_size, AUGMULT, INPUT_SIZE],
    )
    rng = next(prng_seq)
    labels = jax.random.randint(
        rng,
        [NUM_DEVICES, local_batch_size, AUGMULT],
        minval=0,
        maxval=NUM_CLASSES,
    )
    labels = jax.nn.one_hot(labels, NUM_CLASSES)
    batches.append(image_data.DataInputs(image=images, label=labels))
  return batches


class UpdaterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(NUM_DEVICES)
    rng = jax.random.PRNGKey(84452)
    self.rng, self.rng_init = jax.random.split(rng, 2)

    self.net = hk.transform_with_state(model_fn)
    self.forward_fn = forward.MultiClassForwardFn(self.net)

  def init_with_updater(self, updater):
    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)[0]
    (
        self.initial_params,
        self.initial_network_state,
        self.initial_opt_state,
        self.initial_step_count,
    ) = updater.init(rng=self.rng_init, inputs=inputs)

  def run_updater(self, updater, data, return_all_params=False):
    """Runs the updater on the data given in argument."""
    params = self.initial_params
    network_state = self.initial_network_state
    opt_state = self.initial_opt_state
    step_count = self.initial_step_count

    all_params = [utils.get_first(params)]
    for inputs in data:
      # Args are donated. Take copies so that we can reuse them.
      (
          params,
          network_state,
          opt_state,
          step_count,
          unused_scalars,
      ) = updater.update(
          params=jax.tree_map(jnp.copy, params),
          network_state=jax.tree_map(jnp.copy, network_state),
          opt_state=jax.tree_map(jnp.copy, opt_state),
          step_count=step_count,
          inputs=inputs,
      )
      all_params.append(utils.get_first(params))

    return all_params if return_all_params else all_params[-1]

  @parameterized.named_parameters(
      ('no_accumulation_no_weight_decay', 1, 0.0),
      ('no_accumulation_with_weight_decay', 1, 1000.0),
      ('with_accumulation_no_weight_decay', 3, 0.0),
      ('with_accumulation_with_weight_decay', 3, 1000.0),
  )
  def test_accumulation(self, num_accumulations, weight_decay):
    batch_size = LOCAL_BATCH_SIZE * NUM_DEVICES * num_accumulations

    batching = batching_module.VirtualBatching(
        batch_size_init=batch_size,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    # When using weight-decay, ensure that it is the dominant term in the update
    # by using a small learning-rate (downweighs the importance of gradients),
    # so that we can detect it.
    learning_rate = 0.1 if not weight_decay else 0.1 / weight_decay

    updater_no_noise = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=1.0,
            rescale_to_unit_norm=False,
            noise_multiplier=0.0,
            vectorize_grad_clipping=True,
        ),
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
        updater_no_noise, data, return_all_params=True)

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
    std = 0.3
    clipping_norm = 0.1
    batch_size = LOCAL_BATCH_SIZE * NUM_DEVICES * num_accumulations

    batching = batching_module.VirtualBatching(
        batch_size_init=batch_size,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    updater_no_noise = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=clipping_norm,
            rescale_to_unit_norm=False,
            noise_multiplier=0.0,
            vectorize_grad_clipping=True,
        ),
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
    for i in range(NUM_TEST_SAMPLES):
      updater_noise = dp_updater.Updater(
          forward_fn=self.forward_fn,
          batching=batching,
          grad_computer=gradients.GradientComputer(
              clipping_norm=clipping_norm,
              rescale_to_unit_norm=False,
              noise_multiplier=std,
              vectorize_grad_clipping=True,
          ),
          rng_seed=i,
          **_standard_updater_kwargs(),
      )
      # Run one pass of the updater over data with noise using rng_iteration.
      params_noise = self.run_updater(updater_noise, data)
      # The difference with params_no_noise should only contain the noise.
      noise_samples.append(
          _flatten_tree(params_noise) - _flatten_tree(params_no_noise))

    noise_samples = jnp.stack(noise_samples)

    std_expected = std * clipping_norm / batch_size

    # Use synthetic noise as a reference to calibrate the precision required
    # to pass the test.
    synthetic_noise = std_expected * jax.random.normal(self.rng,
                                                       noise_samples.shape)

    # Sanity check: synthetic noise passes KS goodness-of-fit test.
    _, p_synthetic = spst.kstest(
        jnp.ravel(synthetic_noise) / std_expected, 'norm')
    self.assertGreater(p_synthetic, 0.05)

    # Run KS goodness-of-fit test on noise introduced by DP-SGD.
    _, p_dpsgd = spst.kstest(
        jnp.ravel(noise_samples) / std_expected, 'norm')
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
    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=clipping_norm,
            rescale_to_unit_norm=False,
            noise_multiplier=0.0,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params = self.run_updater(updater, data)
    initial_params = utils.get_first(self.initial_params)
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
          self.initial_network_state,
          self.rng,
          inputs_single_device.image,
          is_training=True,
      )

      loss_per_sample_per_augmentation = optax.softmax_cross_entropy(
          logits, inputs_single_device.label)

      # Average over the augmult dimension.
      loss_per_sample = jnp.mean(loss_per_sample_per_augmentation, axis=1)

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
    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    train_only_layer = 'mlp/~/linear_1'
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=0.1,
            rescale_to_unit_norm=False,
            noise_multiplier=0.1,
            vectorize_grad_clipping=True,
        ),
        is_trainable=(
            lambda module_name, *args: module_name == train_only_layer),
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)
    params = self.run_updater(updater, data)

    initial_params = utils.get_first(self.initial_params)

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
            lambda x1, x2: jnp.linalg.norm(x1 - x2) > 1e-2,
            lambda x1, x2: 'Failed',
            params_layer,
            initial_params_layer,
        )
    self.assertEqual(count_trainable, 1)
    self.assertEqual(count_frozen, 2)

  @parameterized.parameters(0.01, 0.1, 1.0, 10.0)
  # TODO: explore why 0.01 and 0.1 clipping norms require higher rtol
  def test_rescaling(self, clipping_norm):
    noise_std = 0.1
    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    updater_no_rescaling = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=clipping_norm,
            noise_multiplier=noise_std,
            rescale_to_unit_norm=False,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )

    updater_with_rescaling = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=clipping_norm,
            noise_multiplier=noise_std,
            rescale_to_unit_norm=True,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )

    self.init_with_updater(updater_no_rescaling)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params_with_rescaling = self.run_updater(updater_with_rescaling, data)
    params_no_rescaling = self.run_updater(updater_no_rescaling, data)
    initial_params = utils.get_first(self.initial_params)

    # Invert SGD equation (with lr=1) to find gradients.
    grads_with_rescaling = (
        _flatten_tree(initial_params) - _flatten_tree(params_with_rescaling))
    grads_no_rescaling = (
        _flatten_tree(initial_params) - _flatten_tree(params_no_rescaling))
    grads_manual_rescaling = grads_no_rescaling / clipping_norm

    assert_trees_all_close(grads_with_rescaling, grads_manual_rescaling)

  def test_evaluation(self):
    forward_fn = _ForwardFnWithRng(self.net)

    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )
    updater = dp_updater.Updater(
        forward_fn=forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=None,
            noise_multiplier=None,
            rescale_to_unit_norm=False,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )
    self.init_with_updater(updater)

    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)
    metrics = updater.evaluate(
        self.initial_params,
        self.initial_network_state,
        self.rng,
        inputs[0])

    # The different devices' outputs should arise from different random keys.
    for j in range(1, NUM_DEVICES):
      self.assertNotAlmostEqual(
          metrics.scalars_avg['rng'][0],
          metrics.scalars_avg['rng'][j])

  def test_average_init_takes_copy(self):
    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=None,
            noise_multiplier=None,
            rescale_to_unit_norm=False,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )
    params = jnp.array([[3., 4., 5.]] * NUM_DEVICES)
    avg_init = updater.init_average(params)
    chex.assert_trees_all_close(params, avg_init)

    # Assert that the average survives even when the original is donated.
    params.delete()
    jax.device_get(avg_init)

  def test_no_averaging_on_accumulation_steps(self):
    # 3 accumulation steps per full batch.
    batch_size = 3 * LOCAL_BATCH_SIZE * NUM_DEVICES

    batching = batching_module.VirtualBatching(
        batch_size_init=batch_size,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=None,
            noise_multiplier=None,
            rescale_to_unit_norm=False,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )

    inputs = _test_data(num_batches=6, local_batch_size=LOCAL_BATCH_SIZE)

    params, network_state, opt_state, step_count = updater.init(
        rng=self.rng_init, inputs=inputs[0])
    for step in range(5):
      if step in (1, 2, 4):
        # This is an accumulation-only step.
        # Average update should be a no-op.
        avg_params = jnp.array([[6., 7., 8.]] * NUM_DEVICES)
        new_params = jnp.array([[3., 4., 5.]] * NUM_DEVICES)
        new_avg_params = updater.update_polyak(
            avg_params, new_params, opt_state, start_step=2)
        chex.assert_trees_all_close(avg_params, new_avg_params)

      # Run an update step to ensure that the multi-step optimiser's
      # accumulation step count is updated. This is how the average updater
      # determines whether this is an update step or an accumulation-only step.
      params, network_state, opt_state, step_count, _ = updater.update(
          params, network_state, opt_state, step_count, inputs[1+step])

  def test_ema_on_update_steps(self):
    # 3 accumulation steps per full batch.
    batch_size = 3 * LOCAL_BATCH_SIZE * NUM_DEVICES

    batching = batching_module.VirtualBatching(
        batch_size_init=batch_size,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )
    updater = dp_updater.Updater(
        forward_fn=self.forward_fn,
        batching=batching,
        grad_computer=gradients.GradientComputer(
            clipping_norm=None,
            noise_multiplier=None,
            rescale_to_unit_norm=False,
            vectorize_grad_clipping=True,
        ),
        **_standard_updater_kwargs(),
    )

    inputs = _test_data(num_batches=8, local_batch_size=LOCAL_BATCH_SIZE)

    params, network_state, opt_state, step_count = updater.init(
        rng=self.rng_init, inputs=inputs[0])
    for step in range(7):
      if step in (3, 6):
        # This is an update step.
        avg_params = jnp.array([[6., 9., 4.]] * NUM_DEVICES)
        new_params = jnp.array([[3., 4., 5.]] * NUM_DEVICES)
        expected_new_avg_params = jnp.array([[3.03, 4.05, 4.99]] * NUM_DEVICES)
        new_avg_params = updater.update_ema(
            avg_params, new_params, opt_state, mu=.01, start_step=-50)
        chex.assert_trees_all_close(expected_new_avg_params, new_avg_params)

      # Run an update step to ensure that the multi-step optimiser's
      # accumulation step count is updated. This is how the average updater
      # determines whether this is an update step or an accumulation-only step.
      params, network_state, opt_state, step_count, _ = updater.update(
          params, network_state, opt_state, step_count, inputs[1+step])


if __name__ == '__main__':
  absltest.main()
