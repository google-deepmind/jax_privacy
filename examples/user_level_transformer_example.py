# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law- agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example of user-level DP-SGD for fine-tuning a transformer model.

This example implements user-level differentially private stochastic gradient
descent (DP-SGD) for a transformer model, using the `jax_privacy` library's
native components.

The implementation demonstrates how to use `UserSelectionStrategy` to handle
unbalanced user datasets efficiently, where each user contributes a different
number of examples.

For more details on the user-level DP-SGD algorithm for large language models,
refer to the paper:
Charles et al. (2024), "Fine-Tuning Large Language Models with User-Level
Differential Privacy" (https://arxiv.org/abs/2404.06713).
"""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy.batch_selection import CyclicPoissonSampling
from jax_privacy.batch_selection import UserSelectionStrategy
from jax_privacy.clipping import clipped_grad


_USERS_PER_BATCH = flags.DEFINE_integer(
    'users_per_batch', 4, 'Number of users to select in each batch.'
)
_EXAMPLES_PER_USER = flags.DEFINE_integer(
    'examples_per_user', 2, 'Number of examples to select for each user.'
)
_STEPS = flags.DEFINE_integer('steps', 10, 'Number of training steps.')
_L2_CLIP_NORM = flags.DEFINE_float(
    'l2_clip_norm', 1.0, 'L2 clipping norm for gradients.'
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')


class TransformerDecoder(nn.Module):
  """A minimal Transformer Decoder."""
  vocab_size: int
  embed_dim: int
  num_heads: int
  ff_dim: int

  @nn.compact
  def __call__(self, x, train: bool):
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)
    x = nn.SelfAttention(
        num_heads=self.num_heads, qkv_features=self.embed_dim
    )(x)
    x = nn.Dense(self.ff_dim)(x)
    x = nn.relu(x)
    x = nn.Dense(self.vocab_size)(x)
    return x


def get_synthetic_data(
    num_users: int,
    num_examples_per_user: list[int],
    seq_len: int,
    vocab_size: int,
):
  """Generates synthetic data for the transformer model."""
  data = []
  labels = []
  user_ids = []
  for i in range(num_users):
    user_data = np.random.randint(
        0, vocab_size, size=(num_examples_per_user[i], seq_len)
    )
    user_labels = np.random.randint(
        0, vocab_size, size=(num_examples_per_user[i], seq_len)
    )
    data.append(user_data)
    labels.append(user_labels)
    user_ids.extend([i] * num_examples_per_user[i])
  return np.concatenate(data), np.concatenate(labels), np.array(user_ids)


def main(_):
  # 1. Model & Data
  vocab_size = 1000
  embed_dim = 64
  num_heads = 4
  ff_dim = 256
  seq_len = 32
  num_users = 20
  num_examples_per_user = np.random.randint(1, 10, size=num_users)

  data, labels, user_ids = get_synthetic_data(
      num_users=num_users,
      num_examples_per_user=num_examples_per_user,
      seq_len=seq_len,
      vocab_size=vocab_size,
  )

  model = TransformerDecoder(
      vocab_size=vocab_size,
      embed_dim=embed_dim,
      num_heads=num_heads,
      ff_dim=ff_dim,
  )
  params = model.init(
      jax.random.key(0), jnp.zeros((1, seq_len), dtype=jnp.int32), train=False
  )['params']
  optimizer = optax.adam(_LEARNING_RATE.value)
  opt_state = optimizer.init(params)

  # 2. Batch Selection
  sampling_prob = _USERS_PER_BATCH.value / num_users
  base_strategy = CyclicPoissonSampling(
      sampling_prob=sampling_prob, iterations=_STEPS.value
  )
  user_strategy = UserSelectionStrategy(
      base_strategy=base_strategy,
      examples_per_user_per_batch=_EXAMPLES_PER_USER.value,
  )

  # 3. Training Step & Clipping
  def loss_fn(params, batch_data, batch_labels):
    logits = model.apply({'params': params}, batch_data, train=True)
    one_hot_labels = jax.nn.one_hot(batch_labels, num_classes=vocab_size)
    return jnp.mean(
        optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    )

  # `clipped_grad` wraps the loss function to compute per-user clipped gradients.
  # With `keep_batch_dim=False`, the loss function receives a batch of examples
  # for a single user. The gradient is computed over this batch, and the
  # resulting gradient (which is an average over the user's examples) is
  # clipped. This aligns with the core requirement of user-level DP.
  grad_fn = clipped_grad(
      loss_fn,
      l2_clip_norm=_L2_CLIP_NORM.value,
      batch_argnums=(1, 2),
      keep_batch_dim=False,
  )

  @jax.jit
  def train_step(params, opt_state, batch_data, batch_labels):
    grads = grad_fn(params, batch_data, batch_labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

  # 4. Training Loop
  batch_iterator = user_strategy.batch_iterator(user_ids, rng=0)
  for step, user_batch_indices in enumerate(batch_iterator):
    if user_batch_indices.size == 0:
      print(f"Step {step}: Skipping empty batch.")
      continue

    batch_data = data[user_batch_indices]
    batch_labels = labels[user_batch_indices]
    params, opt_state = train_step(params, opt_state, batch_data, batch_labels)
    print(f"Step {step}: Completed.")

  print("Training finished successfully.")


if __name__ == '__main__':
  app.run(main)
