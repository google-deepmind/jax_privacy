# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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
from jax_privacy.src.training import batching as batching_module
from jax_privacy.src.training.image_classification import forward
from jax_privacy.src.training.image_classification import updater as updater_module
from jaxline import utils
import numpy as np
import optax
import scipy.stats as spst


INPUT_SIZE = 3
LOCAL_BATCH_SIZE = 2
NUM_CLASSES = 7
NUM_DEVICES = 4
AUGMULT = 5
NUM_TEST_SAMPLES = 28


def get_updater_kwargs():
  return {
      'weight_decay': 0.0,
      'optimizer_name': 'sgd',
      'optimizer_kwargs': {},
      'lr_init_value': 1.0,
      'lr_decay_schedule_name': None,
      'lr_decay_schedule_kwargs': None,
  }


def _flatten_tree(tree):
  return jnp.concatenate([jnp.ravel(x) for x in jax.tree_leaves(tree)])


def model_fn(inputs, is_training=False):
  del is_training  # unused
  return hk.nets.MLP(output_sizes=[INPUT_SIZE, 10, NUM_CLASSES])(inputs)


def assert_close_calibrated(value, ref, expected, rtol=2, atol=1e-5):
  """Check that |value - expected| <= rtol * max(|ref - expected|) + atol."""
  delta = jnp.abs(value - expected)
  max_delta_allowed = rtol * jnp.max(jnp.abs(ref - expected)) + atol
  np.testing.assert_array_less(delta, max_delta_allowed)


def assert_trees_all_close(tree_1, tree_2, rtol=1e-3, atol=1e-5):
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
    batches.append({'images': images, 'labels': labels})
  return batches


class UpdaterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    chex.set_n_cpu_devices(NUM_DEVICES)
    rng = jax.random.PRNGKey(84452)
    self.rng, self.rng_init = jax.random.split(rng, 2)

    self.initial_global_step = jnp.zeros([NUM_DEVICES])

    self.net = hk.transform_with_state(model_fn)
    self.forward_fn = forward.MultiClassForwardFn(self.net)

  def init_with_updater(self, updater):
    inputs = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)[0]
    rng_init = utils.bcast_local_devices(self.rng_init)

    self.initial_params, self.initial_network_state, self.initial_opt_state = (
        updater.init(inputs=inputs, rng_key=rng_init)
    )

  def run_updater(self, updater, data, rng):
    """Runs the updater on the data given in argument."""
    params = self.initial_params
    network_state = self.initial_network_state
    opt_state = self.initial_opt_state
    global_step = self.initial_global_step

    for inputs in data:
      rng_update, rng = jax.random.split(rng)
      params, network_state, opt_state, unused_scalars = updater.update(
          params=params,
          network_state=network_state,
          opt_state=opt_state,
          global_step=global_step,
          inputs=inputs,
          rng=utils.bcast_local_devices(rng_update),
      )
      global_step += 1

    return utils.get_first(params)

  @parameterized.named_parameters(
      ('no_acummulation', 1),
      ('with_acummulation', 5),
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

    updater_noise = updater_module.Updater(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        batching=batching,
        noise_std_relative=std,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=None,
        **get_updater_kwargs(),
    )
    updater_no_noise = updater_module.Updater(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        batching=batching,
        noise_std_relative=0.0,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=None,
        **get_updater_kwargs(),
    )

    self.init_with_updater(updater_noise)

    data = _test_data(
        num_batches=num_accumulations,
        local_batch_size=LOCAL_BATCH_SIZE,
    )

    # Run one pass of the updater over the data with no noise.
    rng_no_noise, rng_noise = jax.random.split(self.rng)
    params_no_noise = self.run_updater(updater_no_noise, data, rng_no_noise)

    # Multiple realizations for different rngs.
    noise_samples = []
    for _ in range(NUM_TEST_SAMPLES):
      # Specialize rng to this iteration.
      rng_iteration, rng_noise = jax.random.split(rng_noise)
      # Run one pass of the updater over data with noise using rng_iteration.
      params_noise = self.run_updater(updater_noise, data, rng_iteration)
      # The difference with params_no_noise should only contain the noise.
      noise_samples.append(
          _flatten_tree(params_noise) - _flatten_tree(params_no_noise))

    noise_samples = jnp.stack(noise_samples)

    std_expected = std * clipping_norm / batch_size

    # Use synthetic noise as a reference to calibrate the precision required
    # to pass the test.
    synthetic_noise = std_expected * jax.random.normal(rng_noise,
                                                       noise_samples.shape)

    # Sanity check: synthetic noise passes KS goodness-of-fit test.
    _, p_synthetic = spst.kstest(  # pytype: disable=attribute-error
        jnp.ravel(synthetic_noise) / std_expected, 'norm')
    self.assertGreater(p_synthetic, 0.05)

    # Run KS goodness-of-fit test on noise introduced by DP-SGD.
    _, p_dpsgd = spst.kstest(  # pytype: disable=attribute-error
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

    updater = updater_module.Updater(
        clipping_norm=clipping_norm,
        rescale_to_unit_norm=False,
        batching=batching,
        noise_std_relative=0.0,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=None,
        **get_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params = self.run_updater(updater, data, self.rng)
    initial_params = utils.get_first(self.initial_params)
    # Invert SGD equation (with lr=1) to find gradients.
    grads_updater = _flatten_tree(initial_params) - _flatten_tree(params)

    # Only one mini-batch in this test.
    inputs = data[0]
    # Merge two leading dimensions to put all data on a single device.
    inputs_single_device = jax.tree_map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
        inputs,
    )

    def forward_per_sample(p):
      logits, unused_network_state = self.net.apply(
          p,
          self.initial_network_state,
          self.rng,
          inputs_single_device['images'],
          is_training=True,
      )

      loss_per_sample_per_augmentation = optax.softmax_cross_entropy(
          logits, inputs_single_device['labels'])

      # Average over the augmult dimension.
      loss_per_sample = jnp.mean(loss_per_sample_per_augmentation, axis=1)

      # Check that the batch dimension is correct.
      chex.assert_shape(loss_per_sample, [LOCAL_BATCH_SIZE * NUM_DEVICES])

      return loss_per_sample

    def clip_global_norm(tree):
      l2_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_leaves(tree)))
      coeff = jnp.minimum(clipping_norm / l2_norm, 1.0)
      return jax.tree_map(lambda x: x * coeff, tree)

    # Compute Jacobian of the loss function.
    jacobian = jax.jacobian(forward_per_sample)(initial_params)
    # Clip Jacobian per sample.
    jacobian_clipped = jax.vmap(clip_global_norm)(jacobian)
    # Average over samples.
    grads_manual = jax.tree_map(
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

    updater = updater_module.Updater(
        clipping_norm=0.1,
        rescale_to_unit_norm=False,
        batching=batching,
        noise_std_relative=0.1,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=train_only_layer,
        **get_updater_kwargs(),
    )

    self.init_with_updater(updater)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)
    params = self.run_updater(updater, data, self.rng)

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
  def test_rescaling(self, clipping_norm):
    noise_std = 0.1
    batching = batching_module.VirtualBatching(
        batch_size_init=LOCAL_BATCH_SIZE * NUM_DEVICES,
        batch_size_per_device_per_step=LOCAL_BATCH_SIZE,
        scale_schedule=None,
    )

    updater_no_rescaling = updater_module.Updater(
        clipping_norm=clipping_norm,
        noise_std_relative=noise_std,
        rescale_to_unit_norm=False,
        batching=batching,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=None,
        **get_updater_kwargs(),
    )

    updater_with_rescaling = updater_module.Updater(
        clipping_norm=clipping_norm,
        noise_std_relative=noise_std,
        rescale_to_unit_norm=True,
        batching=batching,
        train_init=self.forward_fn.train_init,
        forward=self.forward_fn.train_forward,
        train_only_layer=None,
        **get_updater_kwargs(),
    )

    self.init_with_updater(updater_no_rescaling)

    data = _test_data(num_batches=1, local_batch_size=LOCAL_BATCH_SIZE)

    params_with_rescaling = self.run_updater(
        updater_with_rescaling, data, self.rng)
    params_no_rescaling = self.run_updater(
        updater_no_rescaling, data, self.rng)
    initial_params = utils.get_first(self.initial_params)

    # Invert SGD equation (with lr=1) to find gradients.
    grads_with_rescaling = (
        _flatten_tree(initial_params) - _flatten_tree(params_with_rescaling))
    grads_no_rescaling = (
        _flatten_tree(initial_params) - _flatten_tree(params_no_rescaling))
    grads_manual_rescaling = grads_no_rescaling / clipping_norm

    assert_trees_all_close(grads_with_rescaling, grads_manual_rescaling)


if __name__ == '__main__':
  absltest.main()
