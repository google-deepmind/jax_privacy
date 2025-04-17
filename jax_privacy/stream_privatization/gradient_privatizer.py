# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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

"""Definition of stream data privatizer."""

from typing import NamedTuple, Protocol
import jax
from jax_privacy.dp_sgd import typing


class InitFn(Protocol):

  def __call__(self, params: typing.ParamsT) -> typing.NoiseStateT:
    """Initializes any state required to privatize a gradient stream."""


class PrivatizeFn(Protocol):

  def __call__(
      self,
      *,
      sum_of_clipped_grads: typing.ParamsT,
      noise_state: typing.NoiseStateT,
      prng_key: jax.Array,
  ) -> tuple[typing.ParamsT, typing.NoiseStateT]:
    """Privatizes a single sum of clipped gradients.

    This function takes a sum of clipped gradients (so in particular, an array
    tree with known and batch-size-independent sensitivity), and privatizes this
    value.

    NOTE: This API is subject to change, particularly w.r.t. the prng_key
    parameter.  See b/394128304 for more details.

    Args:
      sum_of_clipped_grads: A sum of clipped gradients as described above.
      noise_state: Any state required for computation of potentially-stateful
        noise.
      prng_key: A PRNG key for generating noise, e.g., jax.random.key(0).

    Returns:
      A privatized version of sum_of_clipped_grads.
    """


class GradientPrivatizer(NamedTuple):
  """Encapsulates stateful privatization of a gradient stream.

  Constructors of this class are responsible for ensuring that the results
  of `init` are consumable as the `noise_state` parameter of
  `privatize`. That is, callers are expected to be able to privatize
  gradients by using the following pattern:

  ```python

  mdl_params = ...
  privatizer = GradientPrivatizer(...)
  noise_state = privatizer.init(mdl_params)
  prng_key = jax.random.key(0)
  for _ in range(num_steps):
    prng_key, sub_key = jax.random.split(prng_key)
    sum_of_clipped_grads = ...
    private_sum_of_clipped_grads, noise_state = privatizer.privatize(
        sum_of_clipped_grads=sum_of_clipped_grads, 
        noise_state=noise_state, 
        prng_key=sub_key
    )
    ...
  ```

  Implementations of both `init` and `privatize` are expected to be fully
  serializable JAX logic; IE, compatible with being jitted and compiled.
  """

  init: InitFn
  privatize: PrivatizeFn
