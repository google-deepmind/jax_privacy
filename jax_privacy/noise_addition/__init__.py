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

"""Public API for the noise_addition module.

This module implements optax.GradientTransformations, which we informally call
`privatizers`, that are responsible for taking clipped + aggregated
gradients and adding noise to them. These noise-addition schemes are *stateful*,
meaning the noise added to one gradient may depend on the noise that was added
to previous gradients in various ways. In the simplest case, where i.i.d.
gaussian noise is added to each gradient, this state is nothing more than a
pseudo-random key, each call to `update` uses this key to generate fresh
noise, and splits it into a new key for future steps.

Example Usage:
  >>> import jax
  >>> privatizer = gaussian_privatizer(stddev=1.0, prng_key=jax.random.key(0))
  >>> model = grad = jax.numpy.zeros(10)
  >>> noise_state = privatizer.init(model)
  >>> for _ in range(4):
  ...   noisy_grad, noise_state = privatizer.update(
  ...     sum_of_clipped_grads=grad, noise_state=noise_state
  ...   )

More powerful privatizers, like those based on matrix factorization have
richer state representations, but this is abstracted away from the user via
the optax.GradientTransformation interface. Different privatizers are fully
swappable with each other using the above pattern with only one line of code
changed.

As optax.GradientTransformations, these privatizers can be composed with other
transformations, via optax.chain(privatizer, optimizer). These transformed
privatizers enjoy the same privacy properties by the post-processing property.
"""

# pylint: disable=g-importing-member

from .additive_privatizers import gaussian_privatizer
from .additive_privatizers import matrix_factorization_privatizer
from .additive_privatizers import SupportedStrategies


__all__ = [
    'SupportedStrategies',
    'gaussian_privatizer',
    'matrix_factorization_privatizer',
]
