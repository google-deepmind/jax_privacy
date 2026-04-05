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

from absl import app
from flax import nnx
import jax
import jax.extend.backend
import jax.numpy as jnp
from jax_privacy import batch_selection
from jax_privacy.experimental import execution_plan
import numpy as np
import optax


# Hyperparameters
BATCH_SIZE = 4
CONTEXT_LENGTH = 8
EMBED_SIZE = 16
NUM_HEADS = 4
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_STEPS = 20
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
EPSILON = 10.0
DELTA = 1e-6
PADDING_MULTIPLE = 8


# Data loading and preparation
def download_data(url: str) -> str:
  """Downloads data from a URL.

  Args:
    url: The URL to download data from.

  Returns:
    The content of the downloaded file as a string.
  """
  with urllib.request.urlopen(url, timeout=10) as response:
    return response.read().decode("utf-8")


def create_tokenizer(text: str) -> Dict[str, int]:
  """Creates a simple character-level tokenizer.

  Args:
    text: The input text to build the vocabulary from.

  Returns:
    A dictionary mapping characters to integer indices.
  """
  chars = sorted(list(set(text)))
  return {ch: i for i, ch in enumerate(chars)}


def encode(text: str, stoi: Dict[str, int], context_length: int) -> np.ndarray:
  """Encodes text into a 2D array of non-overlapping sequences.

  Args:
    text: The text to encode.
    stoi: The string-to-index mapping.
    context_length: The length of each sequence.

  Returns:
    A 2D numpy array of shape (num_sequences, context_length).
  """
  data1d = np.array([stoi[ch] for ch in text], dtype=np.int32)
  num_sequences = len(data1d) // context_length
  return data1d[: num_sequences * context_length].reshape(
      (num_sequences, context_length)
  )


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
            in_features=embed_size, out_features=4 * embed_size, rngs=rngs
        ),
        nnx.Linear(
            in_features=4 * embed_size, out_features=embed_size, rngs=rngs
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
    prng_key: jax.Array,
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
    prng_key: An isolated pseudo-random number generator key.
    graphdef: The static graph definition of the NNX model.
    other: Non-trainable state (excluding RNG counts).

  Returns:
    The scalar loss value.
  """
  # Vmap supplies a single prng_key per example.
  # We inject it into the `other` tree mapping.
  other = jax.tree_util.tree_map(
      lambda x: nnx.RngState(prng_key) if isinstance(x, nnx.RngState) else x,
      other,
      is_leaf=lambda x: isinstance(x, nnx.RngState),
  )
  model = nnx.merge(graphdef, params, other)

  # Standard call without rank normalization
  logits = model(x)

  return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def main(argv: list[str]) -> None:
  """Main training loop."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  device = jax.extend.backend.get_backend().platform
  print(f"Starting DP-SGD Training on {device.upper()}...")

  # Data
  text = download_data(
      "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
      "data/tinyshakespeare/input.txt"
  )
  stoi = create_tokenizer(text)
  vocab_size = len(stoi)
  data = encode(text, stoi, CONTEXT_LENGTH)
  print(
      f"Dataset has {len(data)} sequences of length {CONTEXT_LENGTH},"
      f" {vocab_size} vocab size."
  )

  # Model and optimizer
  model = TransformerLM(
      vocab_size=vocab_size,
      embed_size=EMBED_SIZE,
      context_length=CONTEXT_LENGTH,
      num_heads=NUM_HEADS,
      num_layers=NUM_LAYERS,
      rngs=nnx.Rngs(0),
  )
  optimizer = optax.sgd(LEARNING_RATE)

  # graph and other state.
  graphdef, params, other = nnx.split(model, nnx.Param, ...)
  # Remove the internal RNG state from `other` so we can inject them properly
  other = nnx.split(other, nnx.RngState, ...)[2]

  opt_state = optimizer.init(params)

  # Execution Plan Configuration
  dataset_size = len(data)
  config = execution_plan.BandMFExecutionPlanConfig.default(
      iterations=NUM_STEPS,
      num_bands=1,
      epsilon=None,
      delta=None,
      noise_multiplier=NOISE_MULTIPLIER,
      sampling_prob=BATCH_SIZE / dataset_size,
  )
  plan = config.plan
  privatizer = plan.noise_addition_transform
  noise_state = privatizer.init(params)

  # Configure DP Gradient Clipping
  grad_fn = plan.clipped_grad(
      functools.partial(pure_loss_fn, graphdef=graphdef, other=other),
      batch_argnums=(1, 2),  # x and y are batched
      prng_argnum=3,  # Explicitly vmap the PRNG key per batch example
      return_values=True,  # Return loss values for logging
  )

  @jax.jit(donate_argnums=(0, 1, 4))
  def train_step(
      params: nnx.State,
      opt_state: optax.OptState,
      batch: Tuple[jax.Array, jax.Array],
      prng_key: jax.Array,
      noise_state: Any,
      is_padding_example: jax.Array,
  ) -> Tuple[nnx.State, optax.OptState, Any, jax.Array]:
    """Performs a single training step with DP-SGD.

    Args:
      params: Current model parameters.
      opt_state: Current optimizer state.
      batch: A tuple (x, y) of input and target data.
      prng_key: A pseudorandom number generator key.
      noise_state: Current state of the noise mechanism.
      is_padding_example: Boolean mask indicating padding rows.

    Returns:
      Updated params, opt_state, noise_state, and the mean loss for the batch.
    """
    print(f"DEBUG: Compiling train_step for batch size {batch[0].shape[0]}")
    x, y = batch

    # Compute clipped gradients and per-example loss values
    grads, loss = grad_fn(
        params, x, y, prng_key, is_padding_example=is_padding_example
    )
    mean_loss = loss.values.mean()

    assert all(
        g.shape == p.shape
        for g, p in zip(
            jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(params)
        )
    ), "Gradient shapes must match parameter shapes."

    # Add Privacy Noise
    noisy_grads, noise_state = privatizer.update(grads, noise_state)

    # Apply updates using Optax
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = optax.apply_updates(params, updates)

    # loss is an Aux object containing 'values'
    return params, opt_state, noise_state, mean_loss

  # Training loop
  print(f"Training for {NUM_STEPS} steps...")
  iterator = plan.batch_selection_strategy.batch_iterator(dataset_size)
  prng_key = jax.random.key(42)
  for step, batch_indices in enumerate(iterator):
    idx = batch_selection.pad_to_multiple_of(batch_indices, PADDING_MULTIPLE)
    is_padding_example = idx == -1
    safe_idx = np.where(idx == -1, 0, idx)
    batch_seqs = data[safe_idx]
    x = batch_seqs[:, :-1]
    y = batch_seqs[:, 1:]
    batch = (x, y)

    prng_key, subkey = jax.random.split(prng_key)
    params, opt_state, noise_state, loss = train_step(
        params, opt_state, batch, subkey, noise_state, is_padding_example
    )

    print(f"Step {step + 1}/{NUM_STEPS}, Loss: {loss:.4f}")

  print("Training Complete.")


if __name__ == "__main__":
  app.run(main)
