# Copyright 2023, The jax_privacy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of User-Level DP for a transformer model."""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_privacy import batch_selection

FLAGS = flags.FLAGS

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 20, 'Number of training steps.'
)

class CharacterTokenizer:
  """Minimal character-level tokenizer."""

  def __init__(self, text: str):
    self.chars = sorted(list(set(text)))
    self.vocab_size = len(self.chars)
    self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
    self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

  def encode(self, s: str) -> jax.Array:
    return jnp.array([self.char_to_int[c] for c in s], dtype=jnp.int32)

  def decode(self, arr: jax.Array) -> str:
    return ''.join([self.int_to_char[i] for i in np.asarray(arr)])


class TransformerDecoder(nn.Module):
  """Minimal transformer decoder."""
  vocab_size: int
  num_embed: int
  num_heads: int
  num_layers: int

  @nn.compact
  def __call__(self, x: jax.Array, *, is_training: bool) -> jax.Array:
    # Input embedding
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.num_embed)(x)

    # Positional encoding (sinusoidal)
    seq_len = x.shape[1]
    pos = jnp.arange(seq_len)
    pos_enc = jnp.sin(
        pos[:, None] / (10000 ** (jnp.arange(self.num_embed)[None, :] / self.num_embed))
    )
    x += pos_enc

    # Transformer blocks
    for _ in range(self.num_layers):
      x_norm = nn.LayerNorm()(x)
      x_attn = nn.SelfAttention(num_heads=self.num_heads)(x_norm)
      x = x + nn.Dropout(0.1, deterministic=not is_training)(x_attn)
      x_norm = nn.LayerNorm()(x)
      x_ff = nn.Dense(self.num_embed * 4)(x_norm)
      x_ff = nn.relu(x_ff)
      x_ff = nn.Dense(self.num_embed)(x_ff)
      x = x + nn.Dropout(0.1, deterministic=not is_training)(x_ff)

    # Output layer
    x = nn.LayerNorm()(x)
    logits = nn.Dense(self.vocab_size)(x)
    return logits

def pad_sequences(sequences, max_len):
  padded = np.zeros((len(sequences), max_len), dtype=np.int32)
  mask = np.zeros((len(sequences), max_len), dtype=np.float32)
  for i, seq in enumerate(sequences):
    padded[i, :len(seq)] = seq
    mask[i, :len(seq)] = 1.0
  return jnp.asarray(padded), jnp.asarray(mask)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Setup tokenizer and data
  text = "0123456789abcdef"
  tokenizer = CharacterTokenizer(text)
  
  # Create a dummy dataset where each user has a fixed number of sequences
  num_users = 20
  sequences_per_user = 10
  max_len = 20
  
  user_ids = np.repeat(np.arange(num_users), sequences_per_user)
  
  data = []
  for _ in range(num_users * sequences_per_user):
    seq_len = np.random.randint(10, max_len)
    sequence = ''.join(np.random.choice(tokenizer.chars, size=seq_len))
    data.append(tokenizer.encode(sequence))

  padded_data, mask = pad_sequences(data, max_len)

  # 2. Instantiate model and optimizer
  model = TransformerDecoder(
      vocab_size=tokenizer.vocab_size,
      num_embed=64,
      num_heads=4,
      num_layers=2,
  )
  optimizer = optax.sgd(learning_rate=0.1)

  # 3. Define loss function and ULS gradient function
  def loss_fn(params, sequence, mask, rngs):
    batch = sequence[None, :]
    mask = mask[None, :]
    logits = model.apply(
        {'params': params}, batch[:, :-1], is_training=True, rngs=rngs
    )
    labels = jax.nn.one_hot(batch[:, 1:], tokenizer.vocab_size)
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits) * mask[:, 1:, None])
    return loss / jnp.sum(mask[:, 1:])

  per_example_grad_fn = jax.grad(loss_fn)
  
  def train_step(params, opt_state, user_batch_indices, rng_key):
      user_batch_indices = user_batch_indices.astype(jnp.int32)
      user_batch = padded_data[user_batch_indices]
      user_mask_batch = mask[user_batch_indices]
      
      all_user_grads = []
      for i in range(user_batch.shape[0]):
          user_sequences = user_batch[i]
          user_mask = user_mask_batch[i]
          
          user_grads = []
          for j in range(user_sequences.shape[0]):
              sequence = user_sequences[j]
              s_mask = user_mask[j]
              
              rng_key, dropout_key = jax.random.split(rng_key)
              grad = per_example_grad_fn(params, sequence, s_mask, {'dropout': dropout_key})
              user_grads.append(grad)
              
          avg_grad = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *user_grads)
          all_user_grads.append(avg_grad)
          
      if not all_user_grads:
          return params, opt_state, rng_key

      # This is not a jax array, so we can't vmap over it.
      # We will manually stack them.
      per_user_grads = jax.tree.map(lambda *x: jnp.stack(x), *all_user_grads)

      user_clip_norm = 1.0

      def clip_pytree(pytree, max_norm):
          norm = optax.global_norm(pytree)
          scale = jnp.minimum(1.0, max_norm / norm)
          return jax.tree.map(lambda x: x * scale, pytree)

      clipped_grads = jax.vmap(clip_pytree, in_axes=(0, None))(per_user_grads, user_clip_norm)

      sum_clipped_grads = jax.tree.map(lambda x: jnp.sum(x, axis=0), clipped_grads)

      noise_std = 1.0
      noise_rng, rng_key = jax.random.split(rng_key)
      noise = jax.tree.map(
          lambda x: jax.random.normal(noise_rng, x.shape, x.dtype) * noise_std * user_clip_norm,
          sum_clipped_grads,
      )
      noisy_grads = jax.tree.map(lambda x, n: x + n, sum_clipped_grads, noise)

      updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, rng_key
      
  # 4. Training loop
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, padded_data[0, None, :], is_training=False)['params']
  opt_state = optimizer.init(params)
  
  base_strategy = batch_selection.CyclicPoissonSampling(sampling_prob=0.1, iterations=_NUM_STEPS.value)
  user_strategy = batch_selection.UserSelectionStrategy(
      base_strategy, examples_per_user_per_batch=sequences_per_user
  )
    
  rng, seed = jax.random.split(rng)
  seed = jax.random.randint(seed, shape=(), minval=0, maxval=2**31 - 1).item()
  batch_iterator = user_strategy.batch_iterator(user_ids, seed)
    
  print("Starting training with User-Level DP...")
  for step, user_batch_indices in enumerate(batch_iterator):
      rng, step_rng = jax.random.split(rng)
      params, opt_state, rng = train_step(params, opt_state, user_batch_indices, step_rng)
      print(f"Step {step} completed.")
  
  print("Training finished.")

  # 5. Verification
  print("\n--- Verification ---")
  print("ULS noise multiplier would be smaller than ELS for the same privacy level,")
  print("because the sensitivity is now w.r.t. users, not examples.")
  print("For a detailed comparison, one would need to use a privacy accountant.")
  
  print("\nCore ULS training loop code:")
  print("""
def train_step(params, opt_state, user_batch_indices, rng_key):
    # ... (manual loops for users and sequences)
  """)
if __name__ == '__main__':
  app.run(main)
