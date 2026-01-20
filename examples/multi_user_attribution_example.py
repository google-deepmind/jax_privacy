# Copyright 2026, The jax_privacy Authors.
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
"""Example of data selection for datasets with multi-user attribution.

Literature Reference: Ganesh et al. (2025), "It’s My Data Too: Private ML for
Datasets with Multi-User Training Examples".

--- Implementation Notes ---
In many real-world datasets, like group chats or co-authored documents, a
single data point belongs to multiple users. Standard DP training assumes each
data point belongs to one user. This script implements a pre-processing
selection algorithm to build a "safe" dataset where no single user's privacy
is overly compromised.

We implement the "Greedy with Duplicates" approach (Algorithm 3 in the paper).
This method maximizes the final dataset size, which helps reduce variance and
improve the signal-to-noise ratio during DP training, while maintaining a
strict per-user contribution bound (k_limit).
"""

import collections
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import flax.linen as nn
import optax

from jax_privacy.batch_selection import CyclicPoissonSampling


def get_safe_indices(user_mapping, k_limit):
    """
    Selects a subset of data indices using the "Greedy with Duplicates"
    algorithm, ensuring no user's data is included more than k_limit times.

    Args:
      user_mapping: A dictionary mapping an example index to a list of its owners.
      k_limit: The maximum number of times any single user can contribute data.

    Returns:
      A list of indices (with duplicates) that can be safely used for training.
    """
    # To maximize data utilization, we first sort examples by their cardinality.
    # This prioritizes "cheaper" data points (those with fewer owners) that
    # consume less of the global privacy budget in each step.
    sorted_indices = sorted(
        user_mapping.keys(), key=lambda i: len(user_mapping[i])
    )

    user_counts = collections.defaultdict(int)
    safe_indices = []

    # This multi-pass loop implements the "Greedy with Duplicates" strategy.
    # By allowing duplicates in multiple passes, we increase the total sample
    # size N, which significantly improves the signal-to-noise ratio.
    while True:
        added_in_this_pass = 0
        for i in sorted_indices:
            owners = user_mapping[i]
            # Check if all owners of the example are still under the limit.
            if all(user_counts[owner] < k_limit for owner in owners):
                safe_indices.append(i)
                for owner in owners:
                    user_counts[owner] += 1
                added_in_this_pass += 1

        # If a full pass over the sorted data adds no new examples, we're done.
        if added_in_this_pass == 0:
            break

    return safe_indices


def generate_dummy_data(num_examples, num_users, max_owners):
    """Generates a random dataset and user ownership mapping."""
    print(f"Generating dummy data with {num_examples} examples...")
    # Generate feature vectors (e.g., embeddings)
    features = np.random.rand(num_examples, 64).astype(np.float32)
    # Generate random labels
    labels = np.random.randint(0, 10, size=num_examples).astype(np.int32)

    user_ids = [f"User_{i}" for i in range(num_users)]
    user_mapping = {}
    for i in range(num_examples):
        num_owners = np.random.randint(1, max_owners + 1)
        # Each example is owned by 1 to `max_owners` users.
        user_mapping[i] = np.random.choice(
            user_ids, size=num_owners, replace=False
        ).tolist()

    return (features, labels), user_mapping


class SimpleDenseNet(nn.Module):
    """A simple dense model for demonstration."""
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


def main():
    # --- 1. Setup & Data Generation ---
    k_limit = 15
    num_examples = 1_000
    num_users = 200
    max_owners_per_example = 5
    rng_seed = 42

    (features, labels), user_mapping = generate_dummy_data(
        num_examples, num_users, max_owners_per_example
    )

    # --- 2. The Selection Logic ---
    print(f"\nRunning selection logic with k_limit = {k_limit}...")
    safe_indices = get_safe_indices(user_mapping, k_limit)

    # --- 3. Mandatory Correctness Verification ---
    final_user_counts = collections.defaultdict(int)
    for index in safe_indices:
        for owner in user_mapping[index]:
            final_user_counts[owner] += 1

    max_user_contribution = (
        max(final_user_counts.values()) if final_user_counts else 0
    )

    # Assertion Block: Verify that no user appears more than k_limit times.
    assert max_user_contribution <= k_limit, (
        f"Verification failed! User appeared {max_user_contribution} times."
    )

    print(f"[Verification] Verified k_limit check passed. "
          f"Max user contribution: {max_user_contribution}.")

    utilization_rate = len(safe_indices) / num_examples
    print(f"[Metrics] Original items: {num_examples}")
    print(f"[Metrics] Final items (with duplicates): {len(safe_indices)}")
    print(f"[Metrics] Utilization Rate: {utilization_rate:.2%}")


    # --- 4. Integration with a DP Training Loop ---
    print("\nPreparing for training with selected data...")
    safe_features = features[safe_indices]
    safe_labels = labels[safe_indices]
    num_safe_examples = len(safe_indices)

    # Use CyclicPoissonSampling for Poisson subsampling, a key part of DP-SGD.
    batch_size = 256
    num_train_steps = 20
    
    # In Poisson sampling, sampling_prob is batch_size / num_data
    sampling_strategy = CyclicPoissonSampling(
        sampling_prob=batch_size / num_safe_examples if num_safe_examples > 0 else 0,
        iterations=num_train_steps,
    )
    
    batch_iterator = sampling_strategy.batch_iterator(
        num_examples=num_safe_examples, rng=np.random.default_rng(rng_seed)
    )

    # Initialize a simple Flax model and optimizer
    model = SimpleDenseNet()
    key = jax.random.key(rng_seed)
    params = model.init(key, safe_features[0:1])['params']
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    print(f"Starting dummy training loop for {num_train_steps} steps...")
    for step, batch_indices in enumerate(batch_iterator):
        batch_features = safe_features[batch_indices]
        # A real implementation would compute DP gradients on this batch.
        if (step + 1) == num_train_steps:
            print(f"  Step {step+1}: Consumed batch of size {len(batch_indices)}")
        else:
            print(f"  Step {step+1}: Consumed batch of size {len(batch_indices)}", end='\r')

    print("Example finished successfully.")


if __name__ == "__main__":
    main()