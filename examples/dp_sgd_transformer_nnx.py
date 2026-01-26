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

"""Example of training a Transformer with Differential Privacy using Flax NNX.

This script demonstrates how to integrate Flax NNX (the new functional API for
Flax) with JAX Privacy's gradient clipping and noise addition mechanisms. It
implements a simple character-level Transformer language model trained on the
Tiny Shakespeare dataset using DP-SGD.

Key Features:
  - Uses `flax.nnx` for model definition.
  - Implements the "Exhaustive Split" pattern to separate parameters from
    static graph definition.
  - Handles rank normalization for `jax_privacy.clipping.clipped_grad`
    compatibility.
  - Demonstrates correct value access using `clipped_grad` with auxiliary
    outputs.
"""

import functools
from typing import Dict, Tuple, Any
import urllib.request

from flax import nnx  # pytype: disable=import-error
import jax
import jax.extend.backend
import jax.numpy as jnp
from jax_privacy import noise_addition
from jax_privacy.clipping import clipped_grad
import numpy as np
import optax


# Hyperparameters
BATCH_SIZE = 4
CONTEXT_LENGTH = 8
EMBED_SIZE = 16
NUM_HEADS = 4
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_STEPS = 10
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0


# Data loading and preparation
def download_data(url: str) -> str:
  """Downloads data from a URL.

  Args:
    url: The URL to download data from.

  Returns:
    The content of the downloaded file as a string.
  """
  with urllib.request.urlopen(url, timeout=10) as response:
    return response.read().decode('utf-8')


def create_tokenizer(text: str) -> Dict[str, int]:
  """Creates a simple character-level tokenizer.

  Args:
    text: The input text to build the vocabulary from.

  Returns:
    stoi: A dictionary mapping characters to integer indices.
  """
  chars = sorted(list(set(text)))
  stoi = {ch: i for i, ch in enumerate(chars)}
  return stoi


def encode(text: str, stoi: Dict[str, int]) -> np.ndarray:
  """Encodes text using the tokenizer.

  Args:
    text: The text to encode.
    stoi: The string-to-index mapping.

  Returns:
    A numpy array of integer indices.
  """
  return np.array([stoi[ch] for ch in text], dtype=np.int32)


def get_batch(
    data: np.ndarray, batch_size: int, context_length: int
) -> Tuple[np.ndarray, np.ndarray]:
  """Gets a random batch of data.

  Args:
    data: The entire dataset encoded as integers.
    batch_size: The number of examples in the batch.
    context_length: The length of each sequence.

  Returns:
    A tuple (x, y) where:
      - x: Input sequences of shape (batch_size, context_length).
      - y: Target sequences of shape (batch_size, context_length).
  """
  ix = np.random.randint(len(data) - context_length, size=(batch_size,))
  x = np.stack([data[i : i + context_length] for i in ix])
  y = np.stack([data[i + 1 : i + context_length + 1] for i in ix])
  return x, y


class TransformerBlock(nnx.Module):
  """A single Transformer block."""

  def __init__(
      self,
      embed_size: int,
      num_heads: int,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the TransformerBlock.

    Args:
      embed_size: The dimensionality of the embedding.
      num_heads: The number of attention heads.
      rngs: The random number generators.
    """
    self.attention = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=embed_size,
        qkv_features=embed_size,
        out_features=embed_size,
        rngs=rngs,
    )
    self.ln1 = nnx.LayerNorm(num_features=embed_size, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=embed_size, rngs=rngs)
    self.ffw = nnx.Sequential(
        nnx.Linear(
            in_features=embed_size,
            out_features=4 * embed_size,
            rngs=rngs
        ),
        nnx.Linear(
            in_features=4 * embed_size,
            out_features=embed_size,
            rngs=rngs
        ),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies the TransformerBlock to the input.

    Args:
      x: Input array of shape (seq_len, embed_size).

    Returns:
      Output array of the same shape as input.
    """
    # Add is_causal=True to handle masking automatically
    # Note: nnx.MultiHeadAttention does not support is_causal=True in
    # __init__ or __call__ in this version. Using make_causal_mask instead
    # as requested to remove manual generation.
    mask = nnx.make_causal_mask(x[..., 0])
    x = x + self.attention(x, mask=mask, decode=False)
    x = self.ln1(x)
    x = x + self.ffw(x)
    x = self.ln2(x)
    return x


class TransformerLM(nnx.Module):
  """A Transformer language model."""

  def __init__(
      self,
      vocab_size: int,
      *,
      embed_size: int,
      context_length: int,
      num_heads: int,
      num_layers: int,
      rngs: nnx.Rngs,
  ):
    """Initializes the TransformerLM.

    Args:
      vocab_size: The size of the vocabulary.
      embed_size: The dimensionality of the embedding.
      context_length: The max length of the context window.
      num_heads: The number of attention heads.
      num_layers: The number of transformer blocks.
      rngs: The random number generators.
    """
    self.token_embedding = nnx.Embed(
        num_embeddings=vocab_size, features=embed_size, rngs=rngs
    )
    self.pos_embedding = nnx.Embed(
        num_embeddings=context_length, features=embed_size, rngs=rngs
    )
    self.blocks = nnx.List([
        TransformerBlock(embed_size, num_heads, rngs=rngs)
        for _ in range(num_layers)
    ])
    self.ln_f = nnx.LayerNorm(num_features=embed_size, rngs=rngs)
    self.head = nnx.Linear(
        in_features=embed_size, out_features=vocab_size, rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies the model to the input.

    Args:
      x: Input array of token indices with shape (batch_size, seq_len).

    Returns:
      Logits array of shape (batch_size, seq_len, vocab_size).
    """
    pos = jnp.arange(0, x.shape[1])
    tok_emb = self.token_embedding(x)
    pos_emb = self.pos_embedding(pos)
    x = tok_emb + pos_emb
    for block in self.blocks:
      x = block(x)
    x = self.ln_f(x)
    logits = self.head(x)
    return logits


def pure_loss_fn(
    params: nnx.State,
    x: jax.Array,
    y: jax.Array,
    graphdef: nnx.GraphDef,
    other: nnx.State,
) -> jax.Array:
  """A pure functional loss function for DP-SGD.

  This function re-merges the NNX model state, applies the model, and computes
  the cross-entropy loss. It is designed to work with `clipped_grad` which
  requires a functional interface.

  Args:
    params: The trainable parameters of the model.
    x: Input batch (single example or microbatch).
    y: Target batch (single example or microbatch).
    graphdef: The static graph definition of the NNX model.
    other: Non-trainable state (e.g., RNG counts).

  Returns:
    The scalar loss value.
  """
  model = nnx.merge(graphdef, params, other)

  # Standard call without rank normalization
  logits = model(x)


  return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def main():
  """Main training loop."""
  device = jax.extend.backend.get_backend().platform
  print(f"Starting DP-SGD Training on {device.upper()}...")

  # Data
  text = download_data(
      "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
      "data/tinyshakespeare/input.txt"
  )
  stoi = create_tokenizer(text)
  vocab_size = len(stoi)
  data = encode(text, stoi)
  print(f"Dataset has {len(data)} tokens, {vocab_size} vocab size.")

  # Model and optimizer
  rngs = nnx.Rngs(0)
  model = TransformerLM(
      vocab_size=vocab_size,
      embed_size=EMBED_SIZE,
      context_length=CONTEXT_LENGTH,
      num_heads=NUM_HEADS,
      num_layers=NUM_LAYERS,
      rngs=rngs,
  )
  optimizer = optax.adam(LEARNING_RATE)

  # CRITICAL: Exhaustive split to separate trainable params from static
  # graph and other state.
  graphdef, params, other = nnx.split(model, nnx.Param, ...)
  opt_state = optimizer.init(params)

  # Configure DP Gradient Clipping
  grad_fn = clipped_grad(
      functools.partial(pure_loss_fn, graphdef=graphdef, other=other),
      l2_clip_norm=CLIP_NORM,
      batch_argnums=(1, 2),  # x and y are batched
      keep_batch_dim=True,   # Return per-example gradients
      return_values=True,    # Return loss values for logging
      # Note: We do not pass prng_argnum here because 'other' (arg 4) contains
      # RNG state which is handled as a standard argument by NNX.
  )

  privatizer = noise_addition.gaussian_privatizer(
      stddev=grad_fn.sensitivity() * NOISE_MULTIPLIER,
  )
  noise_state = privatizer.init(params)

  @jax.jit
  def train_step(
      params: nnx.State,
      opt_state: optax.OptState,
      batch: Tuple[jax.Array, jax.Array],
      *,
      noise_state: Any,
  ) -> Tuple[nnx.State, optax.OptState, Any, jax.Array]:
    """Performs a single training step with DP-SGD.

    Args:
      params: Current model parameters.
      opt_state: Current optimizer state.
      batch: A tuple (x, y) of input and target data.
      noise_state: Current state of the noise mechanism.

    Returns:
      Updated params, opt_state, noise_state, and the mean loss for the batch.
    """
    x, y = batch

    # Compute clipped gradients and per-example loss values
    grads, loss = grad_fn(params, x, y)

    # Add Privacy Noise
    noisy_grads, noise_state = privatizer.update(grads, noise_state)

    # Apply updates using Optax
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = optax.apply_updates(params, updates)

    # loss is an Aux object containing 'values'
    return params, opt_state, noise_state, loss.values.mean()

  # Training loop
  # TODO: Use jax_privacy.batch_selection.CyclicPoissonSampling
  print(f"Training for {NUM_STEPS} steps...")
  for step in range(NUM_STEPS):
    batch = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH)

    params, opt_state, noise_state, loss = train_step(
        params, opt_state, batch, noise_state=noise_state
    )

    print(f"Step {step + 1}/{NUM_STEPS}, Loss: {loss:.4f}")

  print("Training Complete.")


if __name__ == "__main__":
  main()
