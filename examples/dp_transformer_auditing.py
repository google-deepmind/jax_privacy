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
"""Example of DP-auditing for a transformer model.

This file implements the logic from issue #87 and is intended to serve as a
standalone example for the one-shot transformer auditing suite.
"""

import functools

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jax_privacy
from jax_privacy import noise_addition
from jax_privacy.auditing import CanaryScoreAuditor
from jax_privacy.batch_selection import create_poisson_data_source


FLAGS = flags.FLAGS

_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations', 30, 'Number of training iterations.'
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


def generate_echo_canaries(n, prefix_len, suffix_len, tokenizer):
  """Generates 'echo' canaries with random suffixes."""
  prefix = tokenizer.decode(jnp.arange(min(prefix_len, tokenizer.vocab_size)))
  canaries = set()
  while len(canaries) < 2 * n:
    suffix = ''.join(
        np.random.choice(tokenizer.chars, size=suffix_len, replace=True)
    )
    canaries.add(prefix + suffix)

  canaries = list(canaries)
  canaries_in = canaries[:n]
  canaries_out = canaries[n:]
  return canaries_in, canaries_out


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. Setup tokenizer, data, and canaries
  text = "0123456789abcdef"
  tokenizer = CharacterTokenizer(text)
  canaries_in, canaries_out = generate_echo_canaries(
      n=10, prefix_len=4, suffix_len=4, tokenizer=tokenizer
  )
  data = [tokenizer.encode(s) for s in canaries_in]

  # 2. Instantiate model and optimizer
  model = TransformerDecoder(
      vocab_size=tokenizer.vocab_size,
      num_embed=64,
      num_heads=4,
      num_layers=2,
  )
  optimizer = optax.sgd(learning_rate=0.1)

  # 3. Define loss function and training step
  def loss_fn(params, batch, is_training, rngs=None):
    logits = model.apply(
        {'params': params}, batch[:, :-1], is_training=is_training, rngs=rngs
    )
    labels = jax.nn.one_hot(batch[:, 1:], tokenizer.vocab_size)
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits)) / labels.shape[0]
    return loss, loss  # Return loss twice for has_aux=True

  loss_for_training = functools.partial(loss_fn, is_training=True)
  clipped_grad_fn = jax_privacy.clipped_grad(
      loss_for_training, l2_clip_norm=1.0, has_aux=True
  )

  privatizer = noise_addition.gaussian_privatizer(
      stddev=1.0, prng_key=jax.random.PRNGKey(0)
  )

  # 4. Training loop with auditing
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, data[0][None, :], is_training=False)['params']
  opt_state = optimizer.init(params)
  noise_state = privatizer.init(params)
  scores_in = {}

  data_source = create_poisson_data_source(
      data, sampling_prob=0.1, prng_key=rng
  )

  for i in range(_NUM_ITERATIONS.value):
    batch_indices, batch_data = next(data_source)
    
    if not batch_data:
        continue

    padded_batch = jnp.stack(batch_data)

    rng, dropout_rng_base = jax.random.split(rng)
    dropout_rngs = jax.random.split(
        dropout_rng_base, num=padded_batch.shape[0]
    )
    grads, aux_data = clipped_grad_fn(
        params, padded_batch, rngs={'dropout': dropout_rngs}
    )
    loss = aux_data.aux
    noisy_grads, noise_state = privatizer.update(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
    params = optax.apply_updates(params, updates)


    # Auditor hook: store losses for canaries
    for idx, example_loss in zip(batch_indices, loss):
      if idx < len(canaries_in):
        scores_in[idx] = -example_loss  # Use negative loss as score

    print(f"Iteration {i}, Loss: {jnp.mean(loss)}")

  # 5. Post-training evaluation on `canaries_out`
  loss_for_eval = functools.partial(loss_fn, is_training=False)

  scores_out = [
      -loss_for_eval(params, tokenizer.encode(c)[None, :])[0]
      for c in canaries_out
  ]

  # 6. Statistical Reporting
  auditor = CanaryScoreAuditor(
      in_canary_scores=list(scores_in.values()),
      out_canary_scores=scores_out,
  )
  epsilon_lb = auditor.epsilon_one_run(significance=0.05, delta=1e-5)
  auroc = auditor.attack_auroc()

  print("\n--- Auditing Results ---")
  print(f"Empirical epsilon lower bound: {epsilon_lb:.4f} (delta=1e-5)")
  print(f"Attack AUROC: {auroc:.4f}")


if __name__ == '__main__':
  app.run(main)
