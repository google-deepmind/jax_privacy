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

"""
End-to-end bare-metal JAX Privacy Core API example.

This script demonstrates how to train a simple Transformer model with DP-SGD
using the core, low-level JAX Privacy API. It serves as a minimal, easily
forkable example for users who want to integrate JAX Privacy into a raw JAX
and Flax training loop, without using a higher-level framework like Keras.

Key features of this example:
- A minimal Transformer decoder model defined using Flax Linen.
- A manual training loop in pure JAX.
- Explicit use of the core `jax_privacy` components:
  - `jax_privacy.clipped_grad` for per-example gradient clipping.
  - `jax_privacy.noise_addition.gaussian_privatizer` for noise addition.
  - `jax_privacy.accounting.calibrate` for privacy budget calibration.

This contrasts with the Keras API example (`keras_api_example.py`), which
automates most of these steps within the `make_private` interface.
"""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_privacy
from jax_privacy import keras_api
from jax_privacy import noise_addition
from jax_privacy.accounting import calibrate
from jax_privacy.keras_api import create_poisson_data_source
import numpy as np
import optax


FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for the optimizer.')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training.')
flags.DEFINE_float('target_epsilon', 8.0, 'Target epsilon for DP-SGD.')
flags.DEFINE_float('clipping_norm', 1.0, 'Clipping norm for gradients.')
flags.DEFINE_integer('num_epochs', 1, 'Number of training epochs.')


# --- 1. Data Generation and Tokenization ---


class CharacterTokenizer:
  """A simple character-level tokenizer."""

  def __init__(self, text):
    self.chars = sorted(list(set(text)))
    self.vocab_size = len(self.chars)
    self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
    self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

  def encode(self, text):
    return [self.char_to_id[ch] for ch in text]

  def decode(self, ids):
    return "".join([self.id_to_char[i] for i in ids])





# --- 2. Transformer Model Definition (Flax Linen) ---


class TransformerBlock(nn.Module):
  """A single Transformer block."""

  num_heads: int
  embed_dim: int

  @nn.compact
  def __call__(self, x):
    # Multi-head self-attention
    x_norm = nn.LayerNorm()(x)
    attn_output = nn.SelfAttention(
        num_heads=self.num_heads, qkv_features=self.embed_dim
    )(x_norm)
    x = x + attn_output

    # MLP
    x_norm = nn.LayerNorm()(x)
    mlp_output = nn.Dense(features=self.embed_dim * 4)(x_norm)
    mlp_output = nn.relu(mlp_output)
    mlp_output = nn.Dense(features=self.embed_dim)(mlp_output)
    x = x + mlp_output
    return x


class TransformerDecoder(nn.Module):
  """A minimal Transformer decoder."""

  vocab_size: int
  embed_dim: int
  num_heads: int
  num_layers: int
  seq_len: int

  def setup(self):
    self.blocks = [
        TransformerBlock(num_heads=self.num_heads, embed_dim=self.embed_dim)
        for _ in range(self.num_layers)
    ]

  @nn.compact
  def __call__(self, x):
    # Token and positional embeddings
    embed_layer = nn.Embed(
        num_embeddings=self.vocab_size, features=self.embed_dim
    )
    token_embed = embed_layer(x)
    pos_embed = self.param(
        "pos_embed",
        nn.initializers.normal(stddev=0.02),
        (1, self.seq_len, self.embed_dim),
    )
    x = token_embed + pos_embed

    # Transformer blocks
    for block in self.blocks:
      x = block(x)

    # Final layer norm and output projection
    x = nn.LayerNorm()(x)
    logits = nn.Dense(features=self.vocab_size)(x)
    return logits


# --- 3. DP-SGD Training Logic ---


def loss_fn(params, model, batch):
  """
  Computes cross-entropy loss for a single example (sequence).

  Note that `jax_privacy.clipped_grad` will internally use `jax.vmap`
  to run this function on a batch of data.
  """
  inputs, targets = batch
  logits = model.apply({"params": params}, inputs)
  return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def main(_):
  # Hyperparameters
  seq_len = 64
  train_steps_per_epoch = 50  # Run for a few steps to verify
  target_delta = 1e-5

  # Dummy data
  dummy_text = "JAX Privacy " * 1000
  tokenizer = CharacterTokenizer(dummy_text)
  encoded_text = np.array(tokenizer.encode(dummy_text))
  x_train, y_train = [], []
  for i in range(0, len(encoded_text) - seq_len, 1):
    x_train.append(encoded_text[i : i + seq_len])
    y_train.append(encoded_text[i + 1 : i + seq_len + 1])
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  train_size = len(x_train)

  # Model initialization
  model = TransformerDecoder(
      vocab_size=tokenizer.vocab_size,
      embed_dim=128,
      num_heads=2,
      num_layers=1,
      seq_len=seq_len,
  )
  key = jax.random.key(0)
  dummy_input = jnp.zeros((1, seq_len), dtype=jnp.int32)
  params = model.init(key, dummy_input)["params"]
  tx = optax.adam(FLAGS.learning_rate)
  opt_state = tx.init(params)

  # --- DP-SGD Setup ---
  print("Calibrating noise multiplier...")
  num_updates = FLAGS.num_epochs * train_steps_per_epoch
  dp_params = keras_api.DPKerasConfig(
      epsilon=FLAGS.target_epsilon,
      delta=target_delta,
      clipping_norm=FLAGS.clipping_norm,
      batch_size=FLAGS.batch_size,
      train_steps=num_updates,
      train_size=train_size,
      gradient_accumulation_steps=1,
  )
  accountant = jax_privacy.accounting.analysis.DpsgdTrainingAccountant(
      dp_accountant_config=(
          jax_privacy.accounting.accountants.PldAccountantConfig()
      )
  )
  noise_multiplier = calibrate.calibrate_noise_multiplier(
      target_epsilon=FLAGS.target_epsilon,
      target_delta=target_delta,
      accountant=accountant,
      batch_sizes=FLAGS.batch_size,
      num_updates=num_updates,
      num_samples=train_size,
  )
  print(
      "Using noise multiplier: "
      f"{noise_multiplier:.4f} for ε={FLAGS.target_epsilon}"
  )

  data_source = create_poisson_data_source(
      x_train, y_train, dp_params=dp_params
  )

  # `clipped_grad` wraps the loss function to compute per-example gradients,
  # clip their L2 norm, and sum them. An "example" here is a full sequence.
  # `batch_argnums=2` tells `clipped_grad` that the `batch` argument to
  # `loss_fn` is the one that contains batches of data to be unstacked and
  # processed one by one.
  clipped_grad_fn = jax_privacy.clipped_grad(
      loss_fn,
      l2_clip_norm=FLAGS.clipping_norm,
      batch_argnums=2,
  )

  privatizer = noise_addition.gaussian_privatizer(
      stddev=FLAGS.clipping_norm * noise_multiplier,
      prng_key=jax.random.key(1),
  )
  noise_state = privatizer.init(params)

  @jax.jit
  def train_step(params, opt_state, noise_state, batch):
    inputs, targets, _sample_weight, mask = batch
    # 1. Compute clipped per-example gradients and sum them.
    sum_clipped_grads = clipped_grad_fn(
        params, model, (inputs, targets), is_padding_example=mask
    )

    # 2. Average the gradients.
    avg_grads = jax.tree.map(
        lambda x: x / FLAGS.batch_size, sum_clipped_grads
    )

    # 3. Add noise.
    noisy_grads, new_noise_state = privatizer.update(avg_grads, noise_state)

    # 4. Update parameters.
    updates, new_opt_state = tx.update(noisy_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Also compute loss for logging.
    loss_value = loss_fn(params, model, (inputs, targets))

    return new_params, new_opt_state, new_noise_state, loss_value

  # --- Training Loop ---
  print("Starting training...")
  for epoch in range(FLAGS.num_epochs):
    for step in range(train_steps_per_epoch):
      batch = next(data_source)
      params, opt_state, noise_state, loss_value = train_step(
          params, opt_state, noise_state, batch
      )
      if (step + 1) % 5 == 0:
        print(f"Epoch {epoch}, Step {step+1:3d}, Loss: {loss_value:.4f}")

  print("Training finished.")


if __name__ == "__main__":
  app.run(main)

