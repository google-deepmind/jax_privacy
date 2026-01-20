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
"""Fine-Tuning a Transformer with User-Level Differential Privacy.

Literature Reference: Charles et al. (2024), "Fine-Tuning Large Language Models
with User-Level Differential Privacy" (https://arxiv.org/abs/2407.07737)

--- Implementation Notes ---
Standard DP-SGD clips every example. ULS (User-Level Sampling) is different:
it averages all gradients from a single user into a 'user-update' BEFORE clipping.
As noted in Sec 4.2 of the paper, this is much more efficient for LLMs because
it improves the signal-to-noise ratio by reducing the magnitude of the update
before we hit it with the sensitivity clip.

We use a manual for-loop for the user-aggregation because users often have
varying amounts of data (i.e., unbalanced datasets). This avoids complex
padding and masking issues that can arise with JAX's vmap.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, tree_util
from jax.flatten_util import ravel_pytree
import optax
import tensorflow as tf


from jax_privacy.noise_addition import gaussian_privatizer


# Disable GPU warnings
tf.config.experimental.set_visible_devices([], "GPU")


class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text_data):
        self.vocab = sorted(list(set(text_data)))
        self.vocab_size = len(self.vocab)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, text):
        return np.array([self.char_to_id[c] for c in text], dtype=np.int32)

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids])


class TransformerDecoder:
    """A simple Transformer Decoder model for demonstration."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.vocab_size = vocab_size

    def init_fn(self, key: jax.Array):
        keys = random.split(key, self.n_layers + 3)
        params = {}
        # Token and position embeddings
        params["token_embedding"] = random.normal(
            keys[0], (self.vocab_size, self.d_model)
        )
        params["pos_embedding"] = random.normal(
            keys[1], (self.max_len, self.d_model)
        )

        # Decoder layers
        for i in range(self.n_layers):
            layer_key = keys[i + 2]
            q_key, k_key, v_key, ff_key1, ff_key2 = random.split(layer_key, 5)
            params[f"layer_{i}_q"] = random.normal(
                q_key, (self.d_model, self.d_model)
            )
            params[f"layer_{i}_k"] = random.normal(
                k_key, (self.d_model, self.d_model)
            )
            params[f"layer_{i}_v"] = random.normal(
                v_key, (self.d_model, self.d_model)
            )
            params[f"layer_{i}_ff1"] = random.normal(
                ff_key1, (self.d_model, 4 * self.d_model)
            )
            params[f"layer_{i}_ff2"] = random.normal(
                ff_key2, (4 * self.d_model, self.d_model)
            )

        # Output layer
        params["output_projection"] = random.normal(
            keys[-1], (self.d_model, self.vocab_size)
        )
        return params

    def apply_fn(self, params, inputs: jnp.ndarray):
        # inputs shape: (seq_len,)
        seq_len = inputs.shape[0]
        # Token and positional embeddings
        token_embed = params["token_embedding"][inputs]
        pos_embed = params["pos_embedding"][:seq_len]
        x = token_embed + pos_embed

        # Causal attention mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        for i in range(self.n_layers):
            # Self-attention
            q = x @ params[f"layer_{i}_q"]
            k = x @ params[f"layer_{i}_k"]
            v = x @ params[f"layer_{i}_v"]

            attn_scores = q @ k.T / jnp.sqrt(self.d_model)
            attn_scores = jnp.where(mask, attn_scores, -1e9)
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            attn_output = attn_weights @ v

            # Add & Norm (simplified)
            x = x + attn_output

            # Feed-forward
            ff_output = jax.nn.relu(x @ params[f"layer_{i}_ff1"])
            ff_output = ff_output @ params[f"layer_{i}_ff2"]

            # Add & Norm (simplified)
            x = x + ff_output

        # Final projection
        logits = x @ params["output_projection"]
        return logits


def loss_fn(params, batch):
    """Cross-entropy loss for language modeling."""
    inputs, targets = batch[:-1], batch[1:]
    logits = model.apply_fn(params, inputs)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=-1))


def user_level_grad_fn(params, user_data, user_clip_norm):
    """Averages gradients for one user then clips."""
    num_sequences = len(user_data)
    # Start with a flat zero vector to avoid pytree overhead in the loop
    acc_grad_vector, unravel_fn = ravel_pytree(
        tree_util.tree_map(jnp.zeros_like, params)
    )

    for sequence in user_data:
        # Compute standard grad for one sentence
        grad_pytree = jax.grad(loss_fn)(params, sequence)
        grad_vector, _ = ravel_pytree(grad_pytree)
        acc_grad_vector += grad_vector

    # CORE ULS LOGIC: Average the user's total contribution.
    # This improves signal-to-noise ratio and ensures the 'User Sensitivity'
    # is bounded regardless of how many examples the user has.
    avg_grad_vector = acc_grad_vector / num_sequences

    # Clip the averaged user gradient to bound the influence of the person.
    grad_norm = jnp.linalg.norm(avg_grad_vector)
    multiplier = jnp.minimum(1.0, user_clip_norm / (grad_norm + 1e-6))
    clipped_grad_vector = avg_grad_vector * multiplier

    return clipped_grad_vector, unravel_fn, grad_norm


def train_step(params, opt_state, noise_state, user_batch, optimizer, privatizer):
    """Performs one training step with user-level DP."""
    # We start with a zero vector for accumulating the gradients from the batch.
    # The shape is determined by the total number of model parameters.
    _, unravel_fn_dummy = ravel_pytree(params)
    param_count = ravel_pytree(params)[0].size
    final_grad_vector = jnp.zeros(param_count)
    unravel_fn = None

    for u_id, user_data in user_batch.items():
        # Sanity check: Log how many sequences are being averaged for each user.
        print(
            f"  User {u_id} contribution: "
            f"{len(user_data)} sequences averaged into one gradient."
        )

        # Compute the clipped, averaged gradient for the current user.
        clipped_user_grad, unravel_fn, grad_norm = user_level_grad_fn(
            params, user_data, config["user_clip_norm"]
        )

        # Correctness Verification: Log the pre-clip gradient norm.
        # If this value is > user_clip_norm, it will be clipped.
        print(f"  > User {u_id} avg_grad_norm (pre-clip): {grad_norm:.4f}")


        # Accumulate the clipped gradients from all users in the batch.
        final_grad_vector += clipped_user_grad

    # If unravel_fn was not set in the loop (because batch was empty), use dummy.
    if unravel_fn is None:
        unravel_fn = unravel_fn_dummy

    # Convert the final gradient vector back into the model's pytree structure.
    final_grad_pytree = unravel_fn(final_grad_vector)

    # Add noise to the aggregated gradients using the privatizer.
    noisy_grads, noise_state = privatizer.update(
        final_grad_pytree, noise_state
    )

    # Compute and apply updates using the optax optimizer.
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, noise_state


# --- Configuration ---
config = {
    "num_steps": 10,
    "batch_size": 2,  # Number of users per batch
    "learning_rate": 0.05,
    "user_clip_norm": 1.0,
    "noise_multiplier": 1.1,
    "d_model": 64,
    "n_heads": 2,
    "n_layers": 2,
    "max_len": 32,
}

# --- Synthetic Data Generation ---
# Create synthetic data where some users have more data than others.
synthetic_corpus = (
    "This is a simple text. The goal is to learn patterns."
    "Transformers are powerful. User-level privacy is important."
    "Jax is a high-performance numerical computing library."
)
tokenizer = CharacterTokenizer(synthetic_corpus)
encoded_corpus = tokenizer.encode(synthetic_corpus)

# Structure data by user ID. User 'A' has more data than 'B', 'C', or 'D'.
user_data_db = {
    "A": [
        encoded_corpus[i : i + config["max_len"]] 
        for i in range(0, 100, config["max_len"]) 
    ],
    "B": [encoded_corpus[20 : 20 + config["max_len"]]],
    "C": [encoded_corpus[40 : 40 + config["max_len"]]],
    "D": [encoded_corpus[60 : 60 + config["max_len"]]],
}
user_ids = list(user_data_db.keys())

# --- Model and Optimizer Initialization ---
rng_key = jax.random.key(42)
model = TransformerDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    max_len=config["max_len"],
)
params = model.init_fn(rng_key)

optimizer = optax.adam(learning_rate=config["learning_rate"])
privatizer = gaussian_privatizer(
    stddev=config["user_clip_norm"] * config["noise_multiplier"],
    prng_key=jax.random.key(43),
)

opt_state = optimizer.init(params)
noise_state = privatizer.init(params)

# --- Training Loop ---
print("Starting user-level DP training...")
for step in range(config["num_steps"]):
    # Sample a batch of users for this step
    sampled_user_ids = np.random.choice(
        user_ids, size=config["batch_size"], replace=False
    )
    user_batch = {u_id: user_data_db[u_id] for u_id in sampled_user_ids}

    print(f"\nStep {step+1}/{config['num_steps']}:")
    params, opt_state, noise_state = train_step(
        params, opt_state, noise_state, user_batch, optimizer, privatizer
    )

    # For demonstration, calculate loss on a fixed batch
    fixed_batch = user_data_db["A"][0]
    current_loss = loss_fn(params, fixed_batch)
    print(f"  Loss on fixed batch: {current_loss:.4f}")

print("\nTraining finished.")
