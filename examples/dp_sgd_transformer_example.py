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

"""Example of training a Transformer on text using JAX Privacy core API.

This example demonstrates how to train a simple Transformer decoder
on character-level language modeling using differentially private stochastic
gradient descent (DP-SGD) with the JAX Privacy core library components. It shows how to:
"""

from absl import app
import dp_accounting
import jax
from jax import random
import jax.numpy as jnp
import jax_privacy
from jax_privacy import noise_addition
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


def init_model_params(key, vocab_size=256, embed_dim=128, num_heads=4, num_layers=2, max_len=128):
  """Initializes Transformer parameters."""
  keys = random.split(key, 5)
  model_params = {
      'embedding': random.normal(keys[0], (vocab_size, embed_dim)) * 0.1,
      'pos_embedding': random.normal(keys[1], (max_len, embed_dim)) * 0.1,
      'layers': [],
      'ln_f': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
      'head': random.normal(keys[4], (embed_dim, vocab_size)) * 0.1,
  }
  for i in range(num_layers):
    layer_key = random.split(keys[2], 8)
    layer = {
        'attn': {
            'q': random.normal(layer_key[0], (embed_dim, embed_dim)) * 0.1,
            'k': random.normal(layer_key[1], (embed_dim, embed_dim)) * 0.1,
            'v': random.normal(layer_key[2], (embed_dim, embed_dim)) * 0.1,
            'proj': random.normal(layer_key[3], (embed_dim, embed_dim)) * 0.1,
        },
        'mlp': {
            'fc1': random.normal(layer_key[4], (embed_dim, 4 * embed_dim)) * 0.1,
            'fc2': random.normal(layer_key[5], (4 * embed_dim, embed_dim)) * 0.1,
        },
        'ln1': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
        'ln2': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
    }
    model_params['layers'].append(layer)
  return model_params


def transformer_block(model_params, batch_x, mask):
  """Single transformer block."""
  # Multi-head attention
  q = jnp.dot(batch_x, model_params['attn']['q'])
  k = jnp.dot(batch_x, model_params['attn']['k'])
  v = jnp.dot(batch_x, model_params['attn']['v'])

  num_heads = 4
  head_dim = q.shape[-1] // num_heads
  q = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
  k = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)
  v = v.reshape(v.shape[0], v.shape[1], num_heads, head_dim).transpose(0, 2, 1, 3)

  scale = jnp.sqrt(head_dim)
  attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
  attn = jnp.where(mask, attn, -1e9)
  attn = jax.nn.softmax(attn, axis=-1)
  out = jnp.matmul(attn, v)

  out = out.transpose(0, 2, 1, 3).reshape(batch_x.shape[0], batch_x.shape[1], -1)
  out = jnp.dot(out, model_params['attn']['proj'])

  batch_x = batch_x + out
  batch_x = layer_norm(batch_x, model_params['ln1'])

  # MLP
  h = jnp.dot(batch_x, model_params['mlp']['fc1'])
  h = jax.nn.gelu(h)
  h = jnp.dot(h, model_params['mlp']['fc2'])
  batch_x = batch_x + h
  batch_x = layer_norm(batch_x, model_params['ln2'])
  return batch_x


def layer_norm(batch_x, model_params):
  """Layer normalization."""
  mean = jnp.mean(batch_x, axis=-1, keepdims=True)
  var = jnp.var(batch_x, axis=-1, keepdims=True)
  return model_params['scale'] * (batch_x - mean) / jnp.sqrt(var + 1e-5) + model_params['bias']


def model(model_params, batch_x):
  """Transformer forward pass."""
  seq_len = batch_x.shape[1]
  batch_x = model_params['embedding'][batch_x] + model_params['pos_embedding'][:seq_len]

  mask = jnp.tril(jnp.ones((seq_len, seq_len)))
  mask = mask[None, None, :, :]

  for layer in model_params['layers']:
    batch_x = transformer_block(layer, batch_x, mask)

  batch_x = layer_norm(batch_x, model_params['ln_f'])
  logits = jnp.dot(batch_x, model_params['head'])
  return logits


def loss_fn(model_params, batch_x, batch_y):
  """Cross-entropy loss for next token prediction."""
  logits = model(model_params, batch_x)
  logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
  targets = batch_y[:, 1:].reshape(-1)
  return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def load_text_data(max_len=128):
  """Loads and preprocesses text data."""
  # Use a small text dataset
  text = "Hello world! This is a simple example of text for training a transformer with differential privacy. " * 1000
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  char_to_idx = {ch: i for i, ch in enumerate(chars)}
  idx_to_char = {i: ch for i, ch in enumerate(chars)}

  data = [char_to_idx[ch] for ch in text]
  data = jnp.array(data)

  seqs = []
  for i in range(0, len(data) - max_len - 1, max_len // 2):
    seq = data[i:i+max_len+1]  # +1 for target
    seqs.append(seq)

  seqs = jnp.array(seqs)
  return seqs, vocab_size, char_to_idx, idx_to_char


def batch_dataset(data, batch_size, shuffle=True):
  """Creates batched dataset."""
  dataset = tf.data.Dataset.from_tensor_slices(data)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(data))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def update_model_params(model_params, noisy_grads):
  """Updates parameters using gradients."""
  lr=0.001
  return jax.tree.map(lambda p, g: p - lr * g, model_params, noisy_grads)


def generate_text(model_params, seed_text, length=50, char_to_idx=None, idx_to_char=None):
  """Generates text using the trained model."""
  if char_to_idx is None or idx_to_char is None:
    return "Generation not available without vocab"

  batch_x = jnp.array([[char_to_idx.get(c, 0) for c in seed_text]])
  generated = seed_text

  for _ in range(length):
    logits = model(model_params, batch_x)
    next_token = jnp.argmax(logits[0, -1, :])
    next_char = idx_to_char.get(int(next_token), '?')
    generated += next_char
    batch_x = jnp.concatenate([batch_x, jnp.array([[next_token]])], axis=1)
    if batch_x.shape[1] >= 128:  # max_len
      break

  return generated


def main(_):
  # Hyperparameters
  batch_size = 32
  num_epochs = 10
  learning_rate = 0.001
  clipping_norm = 1.0
  use_dp = True
  epsilon = 1.0
  delta = 1e-5
  max_len = 128

  # Load data
  data, vocab_size, char_to_idx, idx_to_char = load_text_data(max_len)
  train_size = len(data)

  key = random.key(42)
  model_params = init_model_params(key, vocab_size=vocab_size, max_len=max_len)

  if use_dp:
    # Set up DP components
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        accountant=accountant,
        batch_sizes=batch_size,
        num_updates=num_epochs * (train_size // batch_size),
        num_samples=train_size,
    )

    noise_rng = random.key(42)

    grad_fn = jax_privacy.clipped_grad(
        loss_fn,
        l2_clip_norm=clipping_norm,
        batch_argnums=(1, 2),
        has_aux=False,
        return_values=True,
    )
    sensitivity = grad_fn.sensitivity(
        dp_accounting.NeighboringRelation.REPLACE_ONE
    )
    privatizer = noise_addition.gaussian_privatizer(
        stddev=noise_multiplier * sensitivity, prng_key=noise_rng
    )
    noise_state = privatizer.init(model_params)

    @jax.jit
    def dp_train_step(model_params, batch_data, noise_state):
      batch_x = batch_data[:, :-1]
      batch_y = batch_data[:, 1:]
      grads, aux = grad_fn(model_params, batch_x, batch_y)
      loss = aux.values.mean()
      mean_grads = jax.tree.map(lambda g: g / batch_size, grads)
      noisy_grads, noise_state = privatizer.update(mean_grads, noise_state)
      model_params = update_model_params(model_params, noisy_grads, learning_rate)
      return model_params, loss, noise_state

  else:
    @jax.jit
    def train_step(model_params, batch_data):
      batch_x = batch_data[:, :-1]
      batch_y = batch_data[:, 1:]
      loss, grads = jax.value_and_grad(loss_fn)(model_params, batch_x, batch_y)
      model_params = update_model_params(model_params, grads, learning_rate)
      return model_params, loss

  print(f"Training Transformer with {'DP' if use_dp else 'no DP'}...")
  for epoch in range(num_epochs):
    train_ds = batch_dataset(data, batch_size, shuffle=True)

    epoch_loss = 0.0
    num_batches = 0
    for batch_data in train_ds:
      batch_data = jnp.asarray(batch_data)

      if use_dp:
        model_params, loss, noise_state = dp_train_step(
          model_params,
          batch_data,
          noise_state
        )
      else:
        model_params, loss = train_step(
          model_params,
          batch_data
        )

      epoch_loss += loss
      num_batches += 1

    avg_loss = epoch_loss / num_batches

    print(f"Epoch {epoch+1:2d}, Loss: {avg_loss:.4f}")

  generated = generate_text(model_params, "Hello", 50, char_to_idx, idx_to_char)
  print(f"\nGenerated text: {generated}")
  print("\nTraining complete!")


if __name__ == "__main__":
  app.run(main)