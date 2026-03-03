"""Demonstrates cryptographically secure DP-SGD using CPU-side randomness.

Hybrid Entropy Architecture: sourcing hardware-level entropy (RDRAND) on the
CPU to mitigate the "Implementation Gap" and injecting it into the JIT-compiled
GPU step.
"""

from typing import Mapping, Tuple

import time

from absl import app
import dp_accounting
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import batch_selection
from jax_privacy.experimental import accounting
import numpy as np
import optax
import randomgen


USERS = 10_000
FEATURES = 100
ITERATIONS = 500
EXPECTED_BATCH_SIZE = 1000
PADDING_MULTIPLE = 32
EPSILON = 1.0
DELTA = 1e-5
LEARNING_RATE = 0.2
L2_CLIP_NORM = 1.0


def elementwise_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Computes element-wise loss for auditing."""
  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  signed_logits = jnp.where(labels, logits, -logits)
  return -jax.nn.log_sigmoid(signed_logits)


def logistic_loss(
    params: Mapping[str, jax.Array],
    feature_matrix: jax.Array,
    labels: jax.Array,
) -> jax.Array:
  """Computes logistic loss."""
  return jnp.mean(elementwise_loss(params, feature_matrix, labels))


def create_benchmark(
    samples: int,
    features: int,
    seed: int = 0,
) -> Tuple[Mapping[str, jax.Array], jax.Array, jax.Array]:
  """Creates a simple logistic regression model and training data."""
  key = jax.random.key(seed)
  data_key, params_key = jax.random.split(key)

  params = {
      'weights': jax.random.normal(params_key, (features,)),
      'bias': jnp.array(0.0),
  }

  feature_matrix = jax.random.normal(data_key, (samples, features))

  logits = jnp.dot(feature_matrix, params['weights']) + params['bias']
  labels = np.random.rand(samples) < jax.nn.sigmoid(logits)

  return params, feature_matrix, labels


def generate_secure_noise(
    prng: np.random.Generator,
    params: Mapping[str, jax.Array],
    stddev: float,
) -> Mapping[str, np.ndarray]:
  """Generates cryptographic noise for gradients."""

  def sample_noise(leaf):
    return prng.standard_normal(leaf.shape, dtype=np.float32) * stddev

  return jax.tree.map(sample_noise, params)


def main(_):
  """Main function."""
  true_params, all_features, all_labels = create_benchmark(USERS, FEATURES)
  init_params = jax.tree.map(jnp.zeros_like, true_params)

  optimizer = optax.sgd(LEARNING_RATE)
  opt_state = optimizer.init(init_params)

  prng = np.random.Generator(randomgen.RDRAND())
  grad_fn = jax_privacy.clipped_grad(
      logistic_loss,
      l2_clip_norm=L2_CLIP_NORM,
      batch_argnums=(1, 2),
      normalize_by=EXPECTED_BATCH_SIZE,
  )

  make_event = lambda sigma: accounting.dpsgd_event(
      sigma, ITERATIONS, sampling_prob=EXPECTED_BATCH_SIZE / USERS
  )
  noise_multiplier = dp_accounting.calibrate_dp_mechanism(
      dp_accounting.pld.PLDAccountant,
      make_event,
      target_epsilon=EPSILON,
      target_delta=DELTA,
  )
  stddev = noise_multiplier * grad_fn.sensitivity()

  @jax.jit
  def train_step(
      params,
      opt_state,
      batch_features,
      batch_labels,
      is_padding_example,
      secure_noise,
  ):
    """Executes a single training step.

    Args:
      params: Current model parameters.
      opt_state: Current optimizer state.
      batch_features: Feature matrix for the batch.
      batch_labels: Labels for the batch.
      is_padding_example: Boolean array indicating padded examples.
      secure_noise: Pre-generated cryptographic noise for the gradients.

    Returns:
      A tuple containing the updated parameters and the updated optimizer state.
    """
    grads = grad_fn(
        params,
        batch_features,
        batch_labels,
        is_padding_example=is_padding_example,
    )
    noisy_grads = jax.tree.map(jnp.add, grads, secure_noise)
    updates, new_opt_state = optimizer.update(noisy_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  params = init_params
  # WARNING: Batch selection and noise generation happen on CPU
  # for cryptographic security.
  strategy = batch_selection.CyclicPoissonSampling(
      sampling_prob=EXPECTED_BATCH_SIZE / USERS, iterations=ITERATIONS
  )
  start_time = time.perf_counter()
  for step, batch_idx in enumerate(
      strategy.batch_iterator(num_examples=USERS, rng=prng)
  ):
    idx = batch_selection.pad_to_multiple_of(batch_idx, PADDING_MULTIPLE)
    is_padding_example = idx == -1
    batch_features = all_features[idx]
    batch_labels = all_labels[idx]

    secure_noise = generate_secure_noise(prng, params, stddev)

    params, opt_state = train_step(
        params,
        opt_state,
        batch_features,
        batch_labels,
        is_padding_example,
        secure_noise,
    )
    if step > 0 and step % 20 == 0:
      loss = logistic_loss(params, all_features, all_labels)
      elapsed = time.perf_counter() - start_time
      print(f'Step {step}, Loss: {loss:.4f}, Elapsed Time (s): {elapsed:.4f}')
      start_time = time.perf_counter()


if __name__ == '__main__':
  app.run(main)
