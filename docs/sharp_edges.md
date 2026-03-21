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

# Sharp Edges

## Handling Variable Batch Sizes

A major differentiator between differentially private training loops and
standard training loops is the need to handle variable batch sizes. Handled
naively, these variable batch sizes could result in a large number of
compilations, and greatly affect the overall training performance. In the
sections below, we outline a few different strategies you can employ to reduce
this cost.

### Definitions: Global Batch Size vs. Minibatch Size vs. Microbatch Size

**(Physical) Microbatch Size** (default = None): The minibatch will be split
up into smaller microbatches of this size, which will be sequentially
fed into the loss and gradient function using jax.lax.scan. Can reduce
memory at increased sequential computation. Can be especially useful
when the size of the batch inputs is small relative to the intermediate
model activations, as in standard language modeling tasks.
This should generally be set as large as possible.

**(Physical) Minibatch Size** (default = None): This is the number of
elements that are grouped together into a single PyTree of input arrays
before computing and clipping the per-example gradients.

**Global Batch Size**: This is the overall batch size used to compute the
gradient for a single update step. It is what matters for privacy
calculations, and should be chosen to balance utility/compute
trade-offs. The global batch may never be materialized in memory, it
will be processed in chunks of size minibatch_size and further broken
down into chunks of size microbatch_size.

All three of microbatch size, minibatch size, and global batch size, may
vary from iteration to iteration depending on the batch selection strategy.
There are several ways one might try to deal with this enumerated below.

### Approach 1: Pay for recompilation.

Our function to compute the value and clipped gradients is compiled with
respect to a fixed minibatch size and microbatch size. If different
values are encountered, we must recompile the function for the new static
shapes. Often the compilation cost is significantly longer than the
training step time, although this cost is typically amortized over many
training iterations. Recompilation by itself is not a viable approach
unless coupled with a strategy to ensure that the number of different
shapes is relatively small compared to the number of training iterations.

### Approach 2: Pay for padding

While the "physical" minibatch size must be fixed to avoid recompilation
costs, it may consist of "real" and "padding" batch elements, where the
padding batch elements will contribute 0s to the aggregate gradient.
Our function for processing minibatch gradients can ensure that the
contributions from padding elements are zeroed out.

The global batch size can be any multiple of the minibatch size without
paying recompilation costs. Thus, the "cost" of this approach is the
unnecessary compute used to process the padding batch elements, which can
be up to O(minibatch_size). This can be significant depending on how large
the global batch size is relative to the minibatch size.

### Approach 3: Pay for padding + early stopping

We can reduce this cost further from the minibatch size to the microbatch
size, by incorporating dynamic early-stopping once a microbatch with all
padding elements are encountered. Thus, even though the shapes of the
inputs to our compiled function are static, the amount of compute the
function does is dynamic and depends on which batch elements are labeled
as padding examples. This offers a strict improvement over Approach 2, at
the expense of a more complex implementation.

### Approach 4: Truncation + new privacy analysis

In all three approaches above, the global batch size can be any multiple
of the microbatch size without paying for recompilation costs. If we also
require (or desire) a fixed global batch size, then there may be some
chance that the batch size we need is larger than the fixed batch size we
are constrained by, which is problematic. This can be handled by randomly
removing batch elements to reduce to the desired global batch size. This
random dropping of elements requires careful care when doing the privacy
analysis. The cost of this approach is the slack in the privacy analysis
needed to handle truncation, the gap between expected and physical batch
sizes to ensure this occurs with low probability, and relatedly the
processing of padding examples when the sampled batch size is less than
the physical global batch size.

### Approach 5: Hybrid recompilation + padding

Approaches 1, 3, and 4 all have their merits, and the best approach in
terms of compute utilization may vary from setting to setting. These
approaches represent extremes, and a hybrid approach that operates
between them can potentially offer better compute utilization. For
instance, if we allow up to K recompilations for small K, and doing so
greatly reduces the amount of padding batch elements we have to process,
this can be a worthwhile trade-off. Choosing the value of K and the
minibatch/microbatch sizes needed to minimize this cost is the main
challenge to solve here.

## jax.vmap + jax.sharding

When using JAX privacy to train or fine-tune large models with data or model parallelism, one must be careful when imposing sharding annotations on inputs, outputs, and intermediates. Consider the following toy example for demonstration purposes, where we are using pure data parallelism on a simple mean estimation task. The input data consists of a batch of 1024 scalars, and the parameter is a single scalar. The loss function imposes a sharding constraint on the data that it is distributed across these 8 devices. This function (and it’s grad) works great when passing in batches of the expected shape, as demonstrated below:

```python
import jax

jax.config.update('jax_num_cpu_devices', 8)
P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((8,), ('data',), devices=jax.devices())
sharding = jax.sharding.NamedSharding(mesh, P('data'))

def loss(param, data):
  data = jax.lax.with_sharding_constraint(data, sharding)
  return 0.5 * jnp.mean((data - param)**2)

loss_and_grad = jax.grad(loss)

param = 4.0
data = jax.random.normal(jax.random.key(0), (1024,))
loss_and_grad(param, data)
```
```
Array(4.004355, dtype=float32, weak_type=True)
```

However, when we use jax privacy instead, this breaks, as seen below:

```python
import jax_privacy

loss_and_clipped_grad = jax_privacy.clipped_grad(loss, l2_clip_norm=1.0)
loss_and_clipped_grad(param, data)
```
```
ValueError: One of pjit outputs with pytree key path result was given the sharding of NamedSharding(mesh=Mesh('data': 8, axis_types=(Auto,)), spec=PartitionSpec(UNCONSTRAINED, 'data'), memory_kind=unpinned_host), which implies that the global size of its dimension 1 should be divisible by 8, but it is equal to 1 (full shape: (1024, 1))
```

The problem is that `clipped_grad` forms size-1 batches and passes them into
 the loss / grad function using jax.vmap. Since the function expects batches of
 size 1024 (or at least something divisible by 8), this fails. The simplest fix
 is to rewrite loss to remove the sharding annotation (possibly adding it back
 in elsewhere):

```python
def loss(param, data):
  return 0.5 * jnp.mean((data - param)**2)
```

```python
loss_and_clipped_grad = jax_privacy.clipped_grad(loss, l2_clip_norm=jnp.inf, normalize_by=1024)

loss_and_clipped_grad(params, data)
```
```
Array(4.004355, dtype=float32)
```

```python
def loss_and_clipped_grad_wsc(params, data):
  data = jax.lax.with_sharding_constraint(data, sharding)
  return loss(param, data)

loss_and_clipped_grad_wsc(params, data)
```
```
Array(4.004355, dtype=float32)
```

This simple approach may not always be feasible, e.g., if the sharding
constraint is applied to an intermediate value computed by loss rather than an
input/output. One thus be very careful with how the code is designed and how
sharding constraints are incorporated into the program. When trying to integrate
jax privacy into an existing large-scale training framework, this can be
particularly challenging in some cases. The “right” solution to this thorny
issue is not obvious and we do not prescribe a specific approach here, but just
provide a warning to users who might encounter similar issues.

## Mixed precision training

Modern models are often trained with mixed precision, meaning some parameters
and intermediate activations are stored using floating point dtypes of different
levels of precision. Lower precision dtypes like bfloat16 can lead to much
faster array operations and lower memory footprints, but comes at the cost
of decreased precision compared to the standard float32 dtype.

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

| dtype    | time              |
|----------|-------------------|
| float64  | 541ms             |
| float32  | 34.7ms            |
| float16  | 5.58ms            |
| bfloat16 | No Native Support |

Determining what operations can safely be done at lower precision is a crucial
part of the development and scaling process to ensure resources are utilized
effectively. On the other hand, naively using low-precision dtypes
everywhere is known to cause training instabilities, especially at large scales.

The best dtypes strategy to use is in general model specific, although there
are some [general best practices](https://arxiv.org/abs/1710.03740) one can
follow. When using JAX Privacy for training, our recommendations are as follows:

1. Parameters, per-example-gradients, and activations can use whatever
dtype strategy is stable without privacy. i.e., no changes to the loss function
should be needed.
2. Per-example clipped gradients should be accumulated using at least a
precision of float32. This can be accomplished by specifying the `dtype` keyword
arg of [`jax_privacy.clipped_grad`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.clipped_grad.html#jax_privacy.clipping.clipped_grad).
3. Noise should be added using at least a precision of float32. This can be
accomplished by ensuring the input gradients are passed as `float32`, or
specifying the `dtype` arg to the appropriate function from
 [`jax_privacy.noise_addition`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.noise_addition.html).

We believe this strategy strikes a good balance between efficiency and
stability. The accumulation of gradients and addition of noise is generally not
the most expensive part of the training step, and hence the performance cost of
using float32 in these places should generally be small.

As a simple example, consider a standard transformer model. The number of FLOPs
required to do a forward/backward pass is typically approximated as :math:`6
\cdot N \cdot D` where :math:`N` is the number of tokens, and :math:`D` is
the total number of parameters. Meanwhile, accumulating gradients across the
batch dimension requires :math:`B \cdot D'` FLOPS, where :math:`B` is the
batch size (such that :math:`N = B \cdot L` where :math:`L` is sequence
length), and :math:`D'` is the number of *trainable* parameters, which is
sometimes the same as :math:`D` (e.g., pretraining, full fine-tuning) and
sometimes much smaller (e.g., parameter-efficient fine-tuning). Either way,
this is a small fraction of the total number of FLOPS needed for the train
step, and hence higher precision dtypes can be used with minimal performance
impact.
