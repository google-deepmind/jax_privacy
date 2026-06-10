<!-- Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Mixed Precision Training

Modern models are often trained with mixed precision, meaning some parameters
and intermediate activations are stored using floating point dtypes of different
levels of precision. Lower precision dtypes like bfloat16 can lead to much
faster array operations and lower memory footprints, but comes at the cost of
decreased precision compared to the standard float32 dtype.

The simple benchmark below reveals how large the speedups can be from using
lower-precision dtypes. While the exact benefit is hardware-specific, the
results below are obtained by running in a Google colab runtime with a T4 GPU.

```python
N = 1024
k = 64

@jax.jit
def f(A):
    return jnp.dot(A, A.T)

for dtype in [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]:
    A = jax.random.normal(jax.random.key(0), (N, N*k), dtype=dtype)
    _ = jax.block_until_ready(f(A))

    %timeit _ = jax.block_until_ready(f(A))
```

dtype    | time
-------- | -----------------
float64  | 541ms
float32  | 34.7ms
float16  | 5.58ms
bfloat16 | No Native Support

Determining what operations can safely be done at lower precision is a crucial
part of the development and scaling process to ensure resources are utilized
effectively. On the other hand, naively using low-precision dtypes everywhere is
known to cause training instabilities, especially at large scales.

The best dtypes strategy to use is in general model specific, although there are
some [general best practices](https://arxiv.org/abs/1710.03740) one can follow.
When using JAX Privacy for training, our recommendations are as follows:

1.  Parameters, per-example-gradients, and activations can use whatever dtype
    strategy is stable without privacy. i.e., no changes to the loss function
    should be needed.

2.  Per-example clipped gradients should be accumulated using at least a
    precision of float32. This can be accomplished by specifying the `dtype`
    keyword arg of
    [`jax_privacy.clipped_grad`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.clipped_grad.html#jax_privacy.clipping.clipped_grad).

3.  Noise should be added using at least a precision of float32. This can be
    accomplished by ensuring the input gradients are passed as `float32`, or
    specifying the `dtype` arg to the appropriate function from
    [`jax_privacy.noise_addition`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.noise_addition.html).

We believe this strategy strikes a good balance between efficiency and
stability. The accumulation of gradients and addition of noise is generally not
the most expensive part of the training step, and hence the performance cost of
using float32 in these places should generally be small.

As a simple example, consider a standard transformer model. The number of FLOPs
required to do a forward/backward pass is typically approximated as $$6 \cdot N
\cdot D$$ where $$N$$ is the number of tokens, and $$D$$ is the total number of
parameters. Meanwhile, accumulating gradients across the batch dimension
requires $$B \cdot D'$$ FLOPS, where $$B$$ is the batch size (such that $$N = B
\cdot L$$ where $$L$$ is sequence length), and $$D'$$ is the number of
*trainable* parameters, which is sometimes the same as $$D$$ (e.g., pretraining,
full fine-tuning) and sometimes much smaller (e.g., parameter-efficient
fine-tuning). Either way, this is a small fraction of the total number of FLOPS
needed for the train step, and hence higher precision dtypes can be used with
minimal performance impact.
