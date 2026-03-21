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

# Library Design

## Numpy vs. Jax

While this library centers around the JAX ecosystem, it heavily utilizes
standard Python and NumPy for specific modules depending on their computational
requirements.

### When we use JAX
JAX is used for the core training logic that runs on hardware accelerators
(TPUs/GPUs) and benefits from XLA compilation (`jit`), automatic differentiation
(`grad`), and vectorization (`vmap`).

- **Gradient Clipping**: As a key part of any training loop, gradient clipping
  is implemented in terms of jax, utilizing it's `grad` and `vmap` transforms.
- ***Noise Addition**: By default, Noise addition is implemented in terms of JAX
  and is often implemented directly in the training loop using jax-native
  pseudo-random number generators. It is possible to also do noise addition
  host-side (in numpy) with a wider variety of random number generators, which
  can be helpful in situations where cryptographic security is necessary.
- **Matrix Factorization**: Finding the optimal matrix factors requires
  solving a numerical optimization problem with gradient-based solvers.
  We use jax to automatically compute gradients of loss
  functions, and JIT-compilation provides a significant benefit for some
  strategies. Moreover, some of our parameterizations benefit significantly from
  running on accelerators (like `dense.py`).

### When we use NumPy
NumPy is used for host-side (CPU) operations that are not the part of a typical
training loop. This code often involves computations that not highly compatible
with JAX's computation model, like dynamic loops, or do not benefit much
from JAX, like single-use functions.

- **Privacy Accounting**: Computing privacy guarantees and calibrating hyper-
  parameters often involves numerical integration or precise combinatorics.
  Because these are evaluated offline or infrequently, there is no need to run
  them on accelerators, and XLA compilation would not provide significant
  benefit.
- **Batch Selection**: Data loading and batching often occurs on CPUs, before
  being fed to the accelerator in the train step. As such, our batch selection
  API is defined with respect to numpy and runs on the CPU as well. Users can
  pass in a custom NumPy `Generator` to control the behavior of the (pseudo)-
  random number generator.
- **Auditing**: Empirical privacy auditing calculates privacy
  bounds based on canary scores distributions. Like accounting, it heavily uses
  statistical functions (`scipy.stats`), numerical optimization
  (`scipy.optimize`), and offline computations that don't need accelerator
  performance.
