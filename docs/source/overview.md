# Overview

The core of the library written in JAX has the following main components:

*   [Accounting](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/accounting):
    logic to do privacy budget accounting, e.g. compute the budget splitting,
    determine noise multiplier based on epsilon and delta, etc.
*   [DP-SGD](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/dp_sgd):
    classes including public API to implement DP-SGD in raw JAX and Flax linen.
    The main class is
    [`GradientComputer`](https://github.com/google-deepmind/jax_privacy/blob/95870e4d6b9999f849f5426ba3dfab82a20a2317/jax_privacy/dp_sgd/gradients.py#L35)
    which implements public methods to calculate the clipped gradients and add
    noise to them.

Then on top of the core library the following backend-specific public APIs are
built:

*   [Keras](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/keras)

These APIs abstract some complexity and reduce the amount of code necessary to
implement DP training at the cost of less flexibility.
