"""Demonstrates cryptographically secure DP-SGD using CPU-side randomness."""

from typing import Mapping, Tuple

from absl import app
import jax
import jax.numpy as jnp
import jax_privacy
import numpy as np
import optax
from numpy.random import Generator
from randomgen import AESCounter


USERS = 1000
FEATURES = 10
BATCH_SIZE = 32
STEPS = 10
LEARNING_RATE = 0.5
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
    generator: Generator,
    params: Mapping[str, jax.Array],
    stddev: float,
) -> Mapping[str, np.ndarray]:
  """Generates cryptographic noise for gradients."""

  def sample_noise(leaf):
    return generator.standard_normal(leaf.shape, dtype=np.float32) * stddev

  return jax.tree_util.tree_map(sample_noise, params)


def main(_):
  """Main function."""
  true_params, all_features, all_labels = create_benchmark(USERS, FEATURES)
  init_params = jax.tree.map(jnp.zeros_like, true_params)

  optimizer = optax.sgd(LEARNING_RATE)
  opt_state = optimizer.init(init_params)

  seed = 42
  generator = Generator(AESCounter(seed))
  stddev = L2_CLIP_NORM / BATCH_SIZE

  grad_fn = jax_privacy.clipped_grad(
      logistic_loss,
      l2_clip_norm=L2_CLIP_NORM,
      batch_argnums=(1, 2),
      normalize_by=BATCH_SIZE,
  )

  @jax.jit
  def train_step(
      params_, opt_state_, batch_features, batch_labels, secure_noise
  ):
    """Executes a single training step.

    Args:
      params_: Current model parameters.
      opt_state_: Current optimizer state.
      batch_features: Feature matrix for the batch.
      batch_labels: Labels for the batch.
      secure_noise: Pre-generated cryptographic noise for the gradients.

    Returns:
      A tuple containing the updated parameters, the updated optimizer state,
      and a null privatizer state.
    """
    grads = grad_fn(params_, batch_features, batch_labels)
    noisy_grads = jax.tree_util.tree_map(jnp.add, grads, secure_noise)
    updates, new_opt_state = optimizer.update(noisy_grads, opt_state_)
    new_params = optax.apply_updates(params_, updates)
    new_privatizer_state = None
    return new_params, new_opt_state, new_privatizer_state

  params = init_params
  # WARNING: Batch selection and noise generation happen on CPU
  # for cryptographic security.
  for step in range(STEPS):
    idx = generator.choice(USERS, size=BATCH_SIZE, replace=False)
    batch_features = all_features[idx]
    batch_labels = all_labels[idx]

    secure_noise = generate_secure_noise(generator, params, stddev)

    params, opt_state, _ = train_step(
        params,
        opt_state,
        batch_features,
        batch_labels,
        secure_noise,
    )
    if step % 2 == 0:
      loss = logistic_loss(params, all_features, all_labels)
      print(f'Step {step}, Loss: {loss:.4f}')


if __name__ == '__main__':
  app.run(main)
