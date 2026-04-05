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
import time
import flax.linen as nn  # pytype: disable=import-error
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_privacy import batch_selection
from jax_privacy.batch_selection import UserSelectionStrategy
from jax_privacy.experimental import execution_plan


# Constants
USERS_PER_BATCH = 4
EXAMPLES_PER_USER = 2
STEPS = 5
L2_CLIP_NORM = 1.0
LEARNING_RATE = 1e-3
EPSILON = 10.0
DELTA = 1e-6
PADDING_MULTIPLE = 8


class TransformerDecoder(nn.Module):
  """A minimal Transformer Decoder."""

  vocab_size: int
  embed_dim: int
  num_heads: int
  ff_dim: int

  @nn.compact
  def __call__(self, x, train: bool):
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)
    x = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.embed_dim)(
        x
    )
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


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
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
  optimizer = optax.adam(LEARNING_RATE)
  opt_state = optimizer.init(params)

  # 2. Batch Selection & Execution Plan
  # Using BandMFExecutionPlanConfig to configure privacy and strategy
  config = execution_plan.BandMFExecutionPlanConfig.default(
      iterations=STEPS,
      num_bands=1,
      epsilon=EPSILON,
      delta=DELTA,
      sampling_prob=USERS_PER_BATCH / num_users,
  )

  # We create a dummy plan to get the strategy and privatizer
  # Note: `clipped_grad` is created later, but plan.make requires it.
  # However, for strategy and privatizer, we can create them separately or use
  # a placeholder. But plan.make() calculates noise based on sensitivity.
  # We need the grad_fn first.

  # 3. Training Step & Clipping
  def loss_fn(params, x, y, prng_key=None):
    del prng_key  # Unused
    logits = model.apply({'params': params}, x, train=True)
    one_hot_labels = jax.nn.one_hot(y, num_classes=vocab_size)
    return jnp.mean(
        optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    )

  # Create Plan
  plan = config.plan
  privatizer = plan.noise_addition_transform
  noise_state = privatizer.init(params)

  grad_fn = plan.clipped_grad(
      loss_fn,
      batch_argnums=(1, 2),
      prng_argnum=3,
      return_values=True,
  )

  # Wrap the plan's strategy with UserSelectionStrategy
  # We assume plan.batch_selection_strategy is compatible
  # (CyclicPoissonSampling)
  user_strategy = UserSelectionStrategy(
      base_strategy=plan.batch_selection_strategy,
      examples_per_user_per_batch=EXAMPLES_PER_USER,
  )

  @jax.jit(donate_argnums=(0, 1, 4))
  def train_step(
      params, opt_state, x, y, noise_state, prng_key, is_padding_example
  ):
    print(f'DEBUG: Compiling train_step for batch size {x.shape[0]}')
    grads, loss = grad_fn(
        params, x, y, prng_key, is_padding_example=is_padding_example
    )

    # Add Privacy Noise (Using plan's privatizer)
    noisy_grads, noise_state = privatizer.update(grads, noise_state)

    updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, noise_state, loss.values.mean()

  # 4. Training Loop
  start_time = time.time()
  batch_iterator = user_strategy.batch_iterator(user_ids, rng=0)
  prng_key = jax.random.key(42)
  for step, user_batch_indices in enumerate(batch_iterator):
    idx = batch_selection.pad_to_multiple_of(
        user_batch_indices, PADDING_MULTIPLE
    )
    is_padding_example = idx[:, 0] == -1
    safe_idx = np.where(idx == -1, 0, idx)
    x = data[safe_idx]
    y = labels[safe_idx]

    prng_key, subkey = jax.random.split(prng_key)
    params, opt_state, noise_state, loss_val = train_step(
        params, opt_state, x, y, noise_state, subkey, is_padding_example
    )
    print(f'Step {step}: Loss: {loss_val:.4f}')

  end_time = time.time()
  print(f'Total Time: {end_time - start_time:.4f} seconds')
  print('Training finished successfully')


if __name__ == '__main__':
  app.run(main)
