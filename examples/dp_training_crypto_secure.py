"""
Example: DP training with cryptographically secure randomness using Jax Privacy and randomgen.
"""
import numpy as np
try:
    import randomgen
except ImportError:
    raise ImportError("randomgen must be installed for this example.")

from jax_privacy.batch_selection import CryptoSecureBatchSelectionStrategy
from jax_privacy.noise_addition import gaussian_privatizer
import jax
import jax.numpy as jnp
import optax

# Dummy dataset and model for demonstration
NUM_EXAMPLES = 1000
BATCH_SIZE = 128
EPOCHS = 2

# Create a cryptographically secure RNG
crypto_rng = randomgen.RandomGenerator(randomgen.Xoshiro256(secure=True))

# Batch selection strategy using cryptographically secure RNG
batch_selector = CryptoSecureBatchSelectionStrategy()
batch_iter = batch_selector.batch_iterator(NUM_EXAMPLES, rng=crypto_rng)

# Dummy model parameters and gradients
params = jnp.zeros((10,))
clipped_grads = jnp.ones((10,))

# DP privatizer with JAX PRNG (for demonstration, can be extended for randomgen noise)
privatizer = gaussian_privatizer(stddev=1.0, prng_key=jax.random.key(0))
noise_state = privatizer.init(params)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    for _ in range(NUM_EXAMPLES // BATCH_SIZE):
        batch_indices = next(batch_iter)
        # Simulate gradient computation and clipping
        # ...
        # Add DP noise
        noisy_grad, noise_state = privatizer.update(
            sum_of_clipped_grads=clipped_grads, noise_state=noise_state
        )
        print(f"Noisy grad: {noisy_grad}")

print("DP training with cryptographically secure randomness complete.")
