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
on character-level language modeling using differentially private
stochastic gradient descent (DP-SGD) with the JAX Privacy core library
components. The training is done using (DP-SGD) with a synthetic
dataset that is generated from the tiny Shakespeare dataset.

Character-level tokenization is employed, mapping each unique character to
an integer ID. This avoids the complexity of subword tokenizers and keeps
the privacy accounting straightforward. Each training example is a
fixed-length character sequence, and the learning task is next-character
prediction. Privacy is defined at the sequence level, meaning DP-SGD
protects the contribution of any single character sequence. A decoder-only
Transformer with learned token embeddings, learned positional embeddings,
multi-head self-attention implemented via `jax.nn.dot_product_attention`,
and feed-forward (MLP) blocks. Per-example gradients are clipped to
a fixed L2 norm and Gaussian noise is added. Additionally poisson sampling
is used for batch selection to align with standard DP-SGD assumptions
and privacy analysis.

The goal of training is to learn to predict the next character in a sequence,
illustrating how JAX Privacy components can be composed in a realistic
NLP setting.
"""

from absl import app
import dp_accounting
import jax
from jax import random
import jax.numpy as jnp
import jax_privacy
from jax_privacy import batch_selection
from jax_privacy import noise_addition
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate
from jax_privacy.experimental import execution_plan
from typing import Any, Mapping, Sequence, Tuple

import optax
import tensorflow_datasets as tfds


def init_model_params(
    key: jax.Array,
    *,
    vocab_size: int = 256,
    embed_dim: int = 128,
    num_layers: int = 2,
    max_len: int = 128
) -> Mapping[str, Any]:
  """Initializes Transformer parameters."""
  keys = random.split(key, 5)
  model_params = {
      'embedding': random.normal(keys[0], (vocab_size, embed_dim)) * 0.1,
      'pos_embedding': random.normal(keys[1], (max_len, embed_dim)) * 0.1,
      'layers': [],
      'ln_f': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
      'head': random.normal(keys[4], (embed_dim, vocab_size)) * 0.1,
  }
  layer_keys = random.split(keys[2], num_layers * 8)
  for i in range(num_layers):
    layer_key = layer_keys[i*8:(i+1)*8]
    layer = {
      'attn': {
          'q': random.normal(layer_key[0], (embed_dim, embed_dim)) * 0.1,
          'k': random.normal(layer_key[1], (embed_dim, embed_dim)) * 0.1,
          'v': random.normal(layer_key[2], (embed_dim, embed_dim)) * 0.1,
          'proj': random.normal(layer_key[3], (embed_dim, embed_dim)) * 0.1,
      },
      'mlp': {
          'fc1': random.normal(
            layer_key[4],
            (embed_dim, 4 * embed_dim)
          ) * 0.1,
          'fc2': random.normal(
            layer_key[5],
            (4 * embed_dim, embed_dim)
          ) * 0.1,
      },
      'ln1': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
      'ln2': {'scale': jnp.ones(embed_dim), 'bias': jnp.zeros(embed_dim)},
    }
    model_params['layers'].append(layer)
  return model_params


def transformer_block(
    model_params: Mapping[str, Any], batch_x: jax.Array, mask: jax.Array
) -> jax.Array:
  """Single transformer block."""
  # Multi-head attention
  q = jnp.dot(batch_x, model_params['attn']['q'])
  k = jnp.dot(batch_x, model_params['attn']['k'])
  v = jnp.dot(batch_x, model_params['attn']['v'])

  num_heads = 4
  head_dim = q.shape[-1] // num_heads
  q = q.reshape(
    q.shape[0],
    q.shape[1],
    num_heads,
    head_dim
  ).transpose(0, 2, 1, 3)
  k = k.reshape(
    k.shape[0],
    k.shape[1],
    num_heads,
    head_dim
  ).transpose(0, 2, 1, 3)
  v = v.reshape(
    v.shape[0],
    v.shape[1],
    num_heads,
    head_dim
  ).transpose(0, 2, 1, 3)

  out = jax.nn.dot_product_attention(q, k, v, mask=mask)

  out = out.transpose(0, 2, 1, 3).reshape(
    batch_x.shape[0],
    batch_x.shape[1],
    -1
  )
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


def layer_norm(
    batch_x: jax.Array, model_params: Mapping[str, Any]
) -> jax.Array:
  """Layer normalization."""
  mean = jnp.mean(batch_x, axis=-1, keepdims=True)
  var = jnp.var(batch_x, axis=-1, keepdims=True)
  return (
    model_params['scale']
    * (batch_x - mean)
    / jnp.sqrt(var + 1e-5)
    + model_params['bias']
  )


def model(model_params: Mapping[str, Any], batch_x: jax.Array) -> jax.Array:
  """Transformer forward pass."""
  seq_len = batch_x.shape[1]
  batch_x = (
    model_params['embedding'][batch_x]
    + model_params['pos_embedding'][:seq_len]
  )

  mask = jnp.tri(seq_len)
  mask = mask[None, None, :, :]

  for layer in model_params['layers']:
    batch_x = transformer_block(layer, batch_x, mask)

  batch_x = layer_norm(batch_x, model_params['ln_f'])
  logits = jnp.dot(batch_x, model_params['head'])
  return logits


def loss_fn(
    model_params: Mapping[str, Any], batch_x: jax.Array, batch_y: jax.Array
) -> jax.Array:
  """Cross-entropy loss for next token prediction."""
  logits = model(model_params, batch_x)
  logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
  targets = batch_y[:, 1:].reshape(-1)
  return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def load_text_data(
    max_len: int = 128,
) -> Tuple[jax.Array, int, Mapping[str, int], Mapping[int, str]]:
  """Loads and preprocesses text data."""
  # Use tiny Shakespeare dataset
  ds = tfds.load('tiny_shakespeare', split='train')
  text = ''.join(
    example['text'].numpy().decode('utf-8')
    for example in ds.take(100)
  )

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


def generate_text(
    model_params: Mapping[str, Any],
    seed_text: str,
    length: int = 50,
    char_to_idx: Mapping[str, int] | None = None,
    idx_to_char: Mapping[int, str] | None = None
  ) -> str:
  """Generates text using the trained model."""
  if char_to_idx is None or idx_to_char is None:
    return 'Generation not available without vocab'

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


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hyperparameters
  batch_size = 32
  num_epochs = 10
  learning_rate = 0.001
  clipping_norm = 1.0
  epsilon = 1.0
  delta = 1e-5
  max_len = 128
  iterations = 500
  expected_batch_size = 1000
  padding_multiple = 32

  # Load data
  data, vocab_size, char_to_idx, idx_to_char = load_text_data(max_len)
  train_size = len(data)

  key = random.key(42)
  model_params = init_model_params(key, vocab_size=vocab_size, max_len=max_len)

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
      normalize_by=batch_size,
  )
  config = execution_plan.BandMFExecutionPlanConfig(
      iterations=iterations,
      num_bands=1,
      epsilon=epsilon,
      delta=delta,
      sampling_prob=expected_batch_size / train_size,
  )
  plan = config.make(grad_fn)

  sensitivity = grad_fn.sensitivity(
      dp_accounting.NeighboringRelation.REPLACE_ONE
  )
  privatizer = noise_addition.gaussian_privatizer(
      stddev=noise_multiplier * sensitivity, prng_key=noise_rng
  )

  optimizer = optax.sgd(learning_rate)
  privatizer = plan.noise_addition_transform

  @jax.jit
  def dp_train_step(
      model_params: Mapping[str, Any],
      batch_data: Tuple[jax.Array, jax.Array],
      is_padding_example: jax.Array,
      noise_state: Any,
      opt_state: Any
  ) -> Tuple[Mapping[str, Any], Any, jax.Array, Any]:
    batch_x, batch_y = batch_data
    grads, aux = grad_fn(
      model_params,
      batch_x,
      batch_y,
      is_padding_example=is_padding_example
    )
    loss = aux.values.mean()
    noisy_grads, noise_state = privatizer.update(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_state, loss, noise_state

  noise_state = privatizer.init(model_params)
  opt_state = optimizer.init(model_params)

  print('Training Transformer with DP-SGD...')

  for step, batch_idx in enumerate(
      plan.batch_selection_strategy.batch_iterator(train_size)
    ):

    idx = batch_selection.pad_to_multiple_of(batch_idx, padding_multiple)
    is_padding_example = idx == -1
    idx = jnp.where(idx == -1, 0, idx)

    batch_data = data[idx]
    batch_x = batch_data[:, :-1]
    batch_y = batch_data[:, 1:]

    model_params, opt_state, _, noise_state = dp_train_step(
        model_params,
        (batch_x, batch_y),
        is_padding_example,
        noise_state,
        opt_state,
    )

    if step % 100 == 0:
      print(f'Step {step}')

  # Generate sample text with a seed from Shakespeare
  seed_text = 'ROMEO:'
  generated = generate_text(
    model_params,
    seed_text, 100,
    char_to_idx,
    idx_to_char
  )
  print(f'\nGenerated text: {generated}')
  print('\nTraining complete!')


if __name__ == '__main__':
  app.run(main)
