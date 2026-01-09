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

"""Simple end-to-end example of DP training of a transformer.

This example demonstrates how to train a simple transformer model using
DP-SGD with the JAX Privacy core library (not the Keras API). The example
trains a small transformer on a synthetic next-token prediction task.

The example is designed to be:
- Simple: Single file, minimal dependencies
- Educational: Shows all key DP-SGD components
- Forkable: Easy to modify for research purposes

Key JAX Privacy components used:
- jax_privacy.clipped_grad: For per-example gradient clipping
- noise_addition.gaussian_privatizer: For adding calibrated noise
- accounting/calibrate: For computing noise multiplier from privacy budget
"""

from absl import app
from absl import flags
import dp_accounting
import jax
from jax import random
import jax.numpy as jnp
import jax_privacy
from jax_privacy import noise_addition
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate

FLAGS = flags.FLAGS
flags.DEFINE_float("target_epsilon", 8.0, "Target privacy epsilon.")
flags.DEFINE_float("target_delta", 1e-5, "Target privacy delta.")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
flags.DEFINE_float("clipping_norm", 1.0, "L2 norm for gradient clipping.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
flags.DEFINE_bool("use_dp", True, "Whether to use differential privacy.")

# Transformer hyperparameters
VOCAB_SIZE = 100
SEQ_LEN = 16
D_MODEL = 32
NUM_HEADS = 2
D_FF = 64
NUM_LAYERS = 2


def init_transformer_params(key):
  """Initialize a simple transformer's parameters.
  
  This is a minimal transformer without positional encodings for simplicity.
  In practice, you would use a more complete implementation.
  """
  params = {}
  keys = random.split(key, 10)
  key_idx = 0
  
  # Token embedding
  params["token_embedding"] = random.normal(
      keys[key_idx], (VOCAB_SIZE, D_MODEL)
  ) * 0.02
  key_idx += 1
  
  # Transformer layers
  for layer_idx in range(NUM_LAYERS):
    layer_params = {}
    
    # Multi-head self-attention
    layer_params["wq"] = random.normal(
        keys[key_idx], (D_MODEL, NUM_HEADS, D_MODEL // NUM_HEADS)
    ) * 0.02
    key_idx += 1
    layer_params["wk"] = random.normal(
        keys[key_idx], (D_MODEL, NUM_HEADS, D_MODEL // NUM_HEADS)
    ) * 0.02
    key_idx += 1
    layer_params["wv"] = random.normal(
        keys[key_idx], (D_MODEL, NUM_HEADS, D_MODEL // NUM_HEADS)
    ) * 0.02
    key_idx += 1
    layer_params["wo"] = random.normal(
        keys[key_idx], (NUM_HEADS, D_MODEL // NUM_HEADS, D_MODEL)
    ) * 0.02
    key_idx += 1
    
    # Layer norm (simplified: just scale and bias)
    layer_params["ln1_scale"] = jnp.ones(D_MODEL)
    layer_params["ln1_bias"] = jnp.zeros(D_MODEL)
    layer_params["ln2_scale"] = jnp.ones(D_MODEL)
    layer_params["ln2_bias"] = jnp.zeros(D_MODEL)
    
    # Feed-forward network
    layer_params["ff1"] = random.normal(keys[key_idx], (D_MODEL, D_FF)) * 0.02
    key_idx += 1
    layer_params["ff1_bias"] = jnp.zeros(D_FF)
    layer_params["ff2"] = random.normal(keys[key_idx], (D_FF, D_MODEL)) * 0.02
    key_idx += 1
    layer_params["ff2_bias"] = jnp.zeros(D_MODEL)
    
    params[f"layer_{layer_idx}"] = layer_params
    keys = random.split(keys[0], 10)
    key_idx = 0
  
  # Output projection (tied with embedding for simplicity)
  params["output_bias"] = jnp.zeros(VOCAB_SIZE)
  
  return params


def layer_norm(x, scale, bias, eps=1e-5):
  """Apply layer normalization."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def attention(x, wq, wk, wv, wo):
  """Multi-head self-attention."""
  batch_size, seq_len, _ = x.shape
  
  # Compute Q, K, V
  q = jnp.einsum("bsd,dhk->bshk", x, wq)
  k = jnp.einsum("bsd,dhk->bshk", x, wk)
  v = jnp.einsum("bsd,dhk->bshk", x, wv)
  
  # Scaled dot-product attention
  d_k = q.shape[-1]
  scores = jnp.einsum("bqhk,bkhk->bhqk", q, k) / jnp.sqrt(d_k)
  
  # Causal mask
  mask = jnp.tril(jnp.ones((seq_len, seq_len)))
  scores = jnp.where(mask == 0, -1e9, scores)
  
  attn_weights = jax.nn.softmax(scores, axis=-1)
  attn_output = jnp.einsum("bhqk,bkhv->bqhv", attn_weights, v)
  
  # Output projection
  output = jnp.einsum("bshv,hvd->bsd", attn_output, wo)
  return output


def feed_forward(x, ff1, ff1_bias, ff2, ff2_bias):
  """Feed-forward network with GELU activation."""
  h = jax.nn.gelu(jnp.dot(x, ff1) + ff1_bias)
  return jnp.dot(h, ff2) + ff2_bias


def transformer_forward(params, input_ids):
  """Forward pass through the transformer.
  
  Args:
    params: Model parameters dictionary.
    input_ids: Input token IDs of shape (batch_size, seq_len).
    
  Returns:
    Logits of shape (batch_size, seq_len, vocab_size).
  """
  # Token embedding
  x = params["token_embedding"][input_ids]
  
  # Transformer layers
  for layer_idx in range(NUM_LAYERS):
    layer = params[f"layer_{layer_idx}"]
    
    # Self-attention with residual
    attn_out = attention(x, layer["wq"], layer["wk"], layer["wv"], layer["wo"])
    x = layer_norm(
        x + attn_out, layer["ln1_scale"], layer["ln1_bias"]
    )
    
    # Feed-forward with residual
    ff_out = feed_forward(
        x, layer["ff1"], layer["ff1_bias"], layer["ff2"], layer["ff2_bias"]
    )
    x = layer_norm(
        x + ff_out, layer["ln2_scale"], layer["ln2_bias"]
    )
  
  # Output projection (tied with token embedding)
  logits = jnp.dot(x, params["token_embedding"].T) + params["output_bias"]
  return logits


def loss_fn(params, input_ids, target_ids):
  """Compute cross-entropy loss for next-token prediction.
  
  Args:
    params: Model parameters.
    input_ids: Input token IDs of shape (batch_size, seq_len).
    target_ids: Target token IDs of shape (batch_size, seq_len).
    
  Returns:
    Scalar loss value.
  """
  logits = transformer_forward(params, input_ids)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  
  # Gather log probs for target tokens
  batch_size, seq_len = target_ids.shape
  target_log_probs = jnp.take_along_axis(
      log_probs, target_ids[:, :, None], axis=-1
  ).squeeze(-1)
  
  return -jnp.mean(target_log_probs)


def generate_synthetic_data(key, num_samples):
  """Generate synthetic next-token prediction data.
  
  Generates sequences where each token is predictable from the previous ones
  (simple pattern: each token is (previous_token + 1) mod vocab_size).
  """
  keys = random.split(key, num_samples)
  
  # Start with random tokens
  start_tokens = random.randint(keys[0], (num_samples, 1), 0, VOCAB_SIZE)
  
  # Generate sequences with a simple pattern
  sequences = [start_tokens]
  for i in range(SEQ_LEN):
    next_token = (sequences[-1] + 1) % VOCAB_SIZE
    sequences.append(next_token)
  
  full_sequences = jnp.concatenate(sequences, axis=1)
  
  # Input is tokens 0 to seq_len-1, target is tokens 1 to seq_len
  input_ids = full_sequences[:, :-1]
  target_ids = full_sequences[:, 1:]
  
  return input_ids, target_ids


def main(_):
  print("=" * 60)
  print("DP-SGD Transformer Training Example")
  print("=" * 60)
  
  # Configuration
  num_epochs = FLAGS.num_epochs
  batch_size = FLAGS.batch_size
  clipping_norm = FLAGS.clipping_norm
  learning_rate = FLAGS.learning_rate
  use_dp = FLAGS.use_dp
  train_size = 1000
  
  # Initialize
  key = random.key(42)
  key, init_key, data_key = random.split(key, 3)
  
  # Generate synthetic data
  print("\nGenerating synthetic data...")
  input_ids, target_ids = generate_synthetic_data(data_key, train_size)
  print(f"  Training samples: {train_size}")
  print(f"  Sequence length: {SEQ_LEN}")
  print(f"  Vocabulary size: {VOCAB_SIZE}")
  
  # Initialize model
  print("\nInitializing transformer...")
  params = init_transformer_params(init_key)
  num_params = sum(p.size for p in jax.tree.leaves(params))
  print(f"  Number of parameters: {num_params:,}")
  print(f"  Model dimension: {D_MODEL}")
  print(f"  Number of layers: {NUM_LAYERS}")
  print(f"  Number of heads: {NUM_HEADS}")
  
  if use_dp:
    # [START DP SETUP]
    print("\nSetting up differential privacy...")
    
    # Calculate noise multiplier from privacy budget
    accountant = analysis.DpsgdTrainingAccountant(
        dp_accountant_config=accountants.PldAccountantConfig()
    )
    num_updates = num_epochs * (train_size // batch_size)
    
    noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=FLAGS.target_epsilon,
        accountant=accountant,
        batch_sizes=batch_size,
        num_updates=num_updates,
        num_samples=train_size,
        target_delta=FLAGS.target_delta,
    )
    print(f"  Target (ε, δ): ({FLAGS.target_epsilon}, {FLAGS.target_delta})")
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Clipping norm: {clipping_norm}")
    
    # Set up clipped gradients
    grad_and_value_fn = jax_privacy.clipped_grad(
        loss_fn,
        l2_clip_norm=clipping_norm,
        batch_argnums=(1, 2),  # input_ids and target_ids are batched
        has_aux=False,
        return_values=True,
    )
    
    # Compute sensitivity for noise calibration
    sensitivity = grad_and_value_fn.sensitivity(
        dp_accounting.NeighboringRelation.REPLACE_ONE
    )
    
    # Set up Gaussian noise addition
    noise_rng = random.key(123)
    privatizer = noise_addition.gaussian_privatizer(
        stddev=noise_multiplier * sensitivity, prng_key=noise_rng
    )
    noise_state = privatizer.init(params)
    # [END DP SETUP]
    
    @jax.jit
    def dp_train_step(params, batch_input, batch_target, noise_state):
      """Single DP-SGD training step."""
      # Compute per-example clipped gradients
      grads, aux_outputs = grad_and_value_fn(params, batch_input, batch_target)
      loss = aux_outputs.values.mean()
      
      # Average gradients over batch
      mean_grads = jax.tree.map(lambda x: x / batch_size, grads)
      
      # Add calibrated noise
      noisy_grads, noise_state = privatizer.update(mean_grads, noise_state)
      
      # SGD update
      params = jax.tree.map(
          lambda p, g: p - learning_rate * g, params, noisy_grads
      )
      return params, loss, noise_state
    
  else:
    @jax.jit
    def train_step(params, batch_input, batch_target):
      """Single non-private training step."""
      loss, grads = jax.value_and_grad(loss_fn)(params, batch_input, batch_target)
      params = jax.tree.map(
          lambda p, g: p - learning_rate * g, params, grads
      )
      return params, loss
  
  # Training loop
  print(f"\n{'=' * 60}")
  print(f"Training {'with DP' if use_dp else 'without DP'}...")
  print(f"{'=' * 60}")
  
  for epoch in range(num_epochs):
    # Shuffle data
    perm = random.permutation(random.key(epoch), train_size)
    input_ids_shuffled = input_ids[perm]
    target_ids_shuffled = target_ids[perm]
    
    epoch_loss = 0.0
    num_batches = train_size // batch_size
    
    for batch_idx in range(num_batches):
      start_idx = batch_idx * batch_size
      end_idx = start_idx + batch_size
      
      batch_input = input_ids_shuffled[start_idx:end_idx]
      batch_target = target_ids_shuffled[start_idx:end_idx]
      
      if use_dp:
        params, loss, noise_state = dp_train_step(
            params, batch_input, batch_target, noise_state
        )
      else:
        params, loss = train_step(params, batch_input, batch_target)
      
      epoch_loss += loss
    
    avg_loss = epoch_loss / num_batches
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
      print(f"Epoch {epoch + 1:3d}/{num_epochs}: Loss = {avg_loss:.4f}")
  
  # Final evaluation
  print(f"\n{'=' * 60}")
  print("Training complete!")
  print(f"Final loss: {avg_loss:.4f}")
  if use_dp:
    print(f"Privacy guarantee: (ε={FLAGS.target_epsilon}, δ={FLAGS.target_delta})-DP")
  print(f"{'=' * 60}")


if __name__ == "__main__":
  app.run(main)
