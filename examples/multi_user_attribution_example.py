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
"""Example of multi-user attribution for DP training."""

from absl import app
from absl import flags
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_privacy.batch_selection import create_poisson_data_source

FLAGS = flags.FLAGS

_K_LIMIT = flags.DEFINE_integer(
    'k_limit', 10, 'Maximum number of items per user.'
)

def get_safe_indices(user_mapping, k_limit):
  """
  Implements the Greedy Bounding algorithm to select safe indices
  for training in a multi-user attribution scenario.
  """
  num_examples = len(user_mapping)
  
  # Find the total number of unique users
  all_users = set()
  for owners in user_mapping:
    all_users.update(owners)
  num_users = max(all_users) + 1 if all_users else 0

  # Prioritize: Sort examples by the number of owners
  sorted_indices = sorted(range(num_examples), key=lambda i: len(user_mapping[i]))

  user_counts = np.zeros(num_users, dtype=np.int32)
  safe_indices = []
  
  while True:
    added_in_pass = 0
    for example_idx in sorted_indices:
      owners = user_mapping[example_idx]
      
      # Check if all owners have budget
      can_add = True
      for user_id in owners:
        if user_counts[user_id] >= k_limit:
          can_add = False
          break
      
      if can_add:
        safe_indices.append(example_idx)
        for user_id in owners:
          user_counts[user_id] += 1
        added_in_pass += 1

    if added_in_pass == 0:
      break
      
  return safe_indices

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 1. The Data Setup
  num_examples = 1000
  num_users = 200
  
  user_mapping = []
  for _ in range(num_examples):
      num_owners = np.random.randint(1, 6)
      owners = np.random.choice(num_users, size=num_owners, replace=False).tolist()
      user_mapping.append(owners)
      
  # 2. The Selection Algorithm
  safe_indices = get_safe_indices(user_mapping, _K_LIMIT.value)
  
  # 3. The Training Integration
  
  # Create a dummy dataset for training
  dummy_data = jnp.arange(num_examples * 10).reshape((num_examples, 10))
  
  # Use only the safe indices for training
  training_data = dummy_data[np.array(safe_indices)]
  
  # Define a simple model
  class SimpleModel(nn.Module):
      @nn.compact
      def __call__(self, x):
          return nn.Dense(features=1)(x)

  model = SimpleModel()
  
  # Simple training loop
  @jax.jit
  def train_step(params, batch):
      def loss_fn(params):
          logits = model.apply({'params': params}, batch)
          # Dummy loss
          return jnp.mean(logits)

      grad = jax.grad(loss_fn)(params)
      # Dummy optimizer
      return jax.tree.map(lambda p, g: p - 0.1 * g, params, grad)

  rng = jax.random.PRNGKey(0)
  params = model.init(rng, training_data[0])['params']
  
  data_source = create_poisson_data_source(
      training_data, sampling_prob=0.1, prng_key=rng
  )

  print("Starting training...")
  for i in range(20):
      _, batch = next(data_source)
      if not batch:
          continue
      params = train_step(params, jnp.stack(batch))
      print(f"Step {i} completed.")
  print("Training finished.")
  
  # 4. Success Metrics
  total_items_selected = len(safe_indices)
  utilization_rate = total_items_selected / num_examples * 100
  
  print("\n--- Success Metrics ---")
  print(f"Total Items Selected: {total_items_selected} / {num_examples}")
  print(f"Utilization Rate: {utilization_rate:.2f}%")
  
  # Verify that no user appears more than k_limit times
  final_user_counts = np.zeros(num_users, dtype=np.int32)
  for idx in safe_indices:
      for user_id in user_mapping[idx]:
          final_user_counts[user_id] += 1
          
  assert np.all(final_user_counts <= _K_LIMIT.value)
  print(f"Verification successful: No user appears more than {_K_LIMIT.value} times.")
  print(f"Max items for any user: {np.max(final_user_counts)}")

if __name__ == '__main__':
  app.run(main)
