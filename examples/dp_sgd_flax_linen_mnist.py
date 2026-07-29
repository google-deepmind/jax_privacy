# coding=utf-8
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

"""DP-SGD training of a Flax Linen CNN on MNIST (jax_privacy 2.x).

Replaces the broken ``dp_sgd_flax_linen_mnist.ipynb`` tutorial, which only
works with ``jax-privacy==1.0`` and the removed ``jax_privacy.dp_sgd`` API.

Pipeline:
  1. Calibrate Gaussian noise via ``accounting.dpsgd_event`` +
     ``dp_accounting.calibrate_dp_mechanism`` (PLD accountant).
  2. Sample batches with ``batch_selection.CyclicPoissonSampling``.
  3. Per-example clip + sum with ``jax_privacy.clipped_grad`` (normalize by
     expected batch size; pad empty Poisson draws).
  4. Add noise with ``noise_addition.gaussian_privatizer``.

Full run (defaults) targets roughly the old notebook setup (~92% test at
ε≈1, δ=1e-5). For a CI smoke::

  python examples/dp_sgd_flax_linen_mnist.py --smoke
"""

from typing import Any, Mapping, Tuple

from absl import app
from absl import flags
import dp_accounting
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import accounting
from jax_privacy import batch_selection
from jax_privacy import noise_addition
import keras
import numpy as np
import optax

_SMOKE = flags.DEFINE_bool(
    'smoke',
    False,
    'CI smoke mode: few steps, no accuracy assertion.',
)
_TRAIN_STEPS = flags.DEFINE_integer(
    'train_steps',
    5000,
    'Number of DP-SGD optimization steps.',
)
_EXPECTED_BATCH_SIZE = flags.DEFINE_integer(
    'expected_batch_size',
    256,
    'Expected Poisson batch size (also used as normalize_by).',
)
_EPSILON = flags.DEFINE_float('epsilon', 1.0, 'Target ε.')
_DELTA = flags.DEFINE_float('delta', 1e-5, 'Target δ.')
_CLIPPING_NORM = flags.DEFINE_float(
    'clipping_norm',
    0.1,
    'Per-example L2 clip norm (before rescale_to_unit_norm).',
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.1, 'SGD learning rate.')
_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'SGD momentum.')
_EVAL_EVERY = flags.DEFINE_integer(
    'eval_every',
    200,
    'Evaluate test accuracy every N steps (and at the end).',
)
_PADDING_MULTIPLE = flags.DEFINE_integer(
    'padding_multiple',
    32,
    'Pad Poisson batches to a multiple of this size (limits JIT shapes).',
)
_SEED = flags.DEFINE_integer('seed', 0, 'PRNG seed.')


class CNN(nn.Module):
  """Simple CNN matching the legacy Flax Linen MNIST notebook."""

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Loads MNIST as float32 images in [0, 1] with integer labels."""
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, -1)
  x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, -1)
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)
  return x_train, y_train, x_test, y_test


def loss_fn(
    params: Mapping[str, Any],
    images: jax.Array,
    labels: jax.Array,
    apply_fn: Any,
) -> jax.Array:
  """Scalar loss; clipped_grad evaluates this per example."""
  logits = apply_fn({'params': params}, images)
  return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def accuracy(
    params: Mapping[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    apply_fn: Any,
    batch_size: int = 512,
) -> float:
  """Computes classification accuracy over a full dataset."""
  correct = 0
  total = images.shape[0]
  for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    logits = apply_fn({'params': params}, jnp.asarray(images[start:end]))
    preds = jnp.argmax(logits, axis=-1)
    correct += int(jnp.sum(preds == jnp.asarray(labels[start:end])))
  return correct / total


def _pad_batch_indices(indices: np.ndarray, multiple: int) -> np.ndarray:
  """Pads indices; empty Poisson draws become an all-padding batch."""
  if indices.size == 0:
    return np.full(multiple, -1, dtype=np.int32)
  return batch_selection.pad_to_multiple_of(indices, multiple)


def main(_):
  train_steps = 50 if _SMOKE.value else _TRAIN_STEPS.value
  eval_every = 25 if _SMOKE.value else _EVAL_EVERY.value
  expected_batch_size = _EXPECTED_BATCH_SIZE.value
  epsilon = _EPSILON.value
  delta = _DELTA.value
  clipping_norm = _CLIPPING_NORM.value
  padding_multiple = _PADDING_MULTIPLE.value
  seed = _SEED.value

  x_train, y_train, x_test, y_test = load_mnist()
  train_size = x_train.shape[0]
  sampling_prob = expected_batch_size / train_size

  model = CNN()
  key = jax.random.key(seed)
  params = model.init(key, jnp.ones([1, 28, 28, 1]))['params']
  apply_fn = model.apply

  def batched_loss(params, images, labels):
    return loss_fn(params, images, labels, apply_fn)

  make_event = lambda sigma: accounting.dpsgd_event(
      sigma, train_steps, sampling_prob=sampling_prob
  )
  noise_multiplier = dp_accounting.calibrate_dp_mechanism(
      dp_accounting.pld.PLDAccountant,
      make_event,
      target_epsilon=epsilon,
      target_delta=delta,
  )

  grad_fn = jax_privacy.clipped_grad(
      batched_loss,
      l2_clip_norm=clipping_norm,
      batch_argnums=(1, 2),
      normalize_by=expected_batch_size,
      rescale_to_unit_norm=True,
  )
  stddev = noise_multiplier * grad_fn.l2_norm_bound
  privatizer = noise_addition.gaussian_privatizer(
      stddev=stddev, prng_key=jax.random.key(seed + 1)
  )

  accountant = dp_accounting.pld.PLDAccountant()
  accountant.compose(make_event(noise_multiplier))
  achieved_epsilon = accountant.get_epsilon(target_delta=delta)

  print(
      f'Calibrated noise_multiplier={noise_multiplier:.4f}, '
      f'stddev={stddev:.6f}, sampling_prob={sampling_prob:.6f}'
  )
  print(
      f'Target (ε, δ)=({epsilon}, {delta}); '
      f'achieved ε={achieved_epsilon:.4f} at δ={delta}'
  )

  optimizer = optax.sgd(_LEARNING_RATE.value, momentum=_MOMENTUM.value)
  opt_state = optimizer.init(params)
  noise_state = privatizer.init(params)

  @jax.jit
  def train_step(
      params,
      opt_state,
      noise_state,
      images,
      labels,
      is_padding_example,
  ):
    clipped_grads = grad_fn(
        params, images, labels, is_padding_example=is_padding_example
    )
    noisy_grads, noise_state = privatizer.update(clipped_grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, noise_state

  strategy = batch_selection.CyclicPoissonSampling(
      sampling_prob=sampling_prob,
      iterations=train_steps,
  )
  rng = np.random.default_rng(seed)

  for step, batch_idx in enumerate(
      strategy.batch_iterator(num_examples=train_size, rng=rng)
  ):
    idx = _pad_batch_indices(batch_idx, padding_multiple)
    is_padding = idx == -1
    safe_idx = np.where(is_padding, 0, idx)
    batch_images = jnp.asarray(x_train[safe_idx])
    batch_labels = jnp.asarray(y_train[safe_idx])

    params, opt_state, noise_state = train_step(
        params,
        opt_state,
        noise_state,
        batch_images,
        batch_labels,
        jnp.asarray(is_padding),
    )

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
      acc = accuracy(params, x_test, y_test, apply_fn)
      print(f'Step {step}: test accuracy={acc * 100:.2f}%')

  test_acc = accuracy(params, x_test, y_test, apply_fn)
  print(f'Final ε={achieved_epsilon:.4f} (δ={delta})')
  print(f'Final test accuracy={test_acc * 100:.2f}%')

  if not _SMOKE.value:
    # Full run should be well above chance; ~92% with default HPs.
    assert test_acc > 0.80, f'Test accuracy {test_acc:.4f} is too low'


if __name__ == '__main__':
  app.run(main)
