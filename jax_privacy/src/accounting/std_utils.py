# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

"""Util functions to adjust the standard deviation."""
import chex
import jax.numpy as jnp


def divide_std_over_sum(
    target_std: chex.Numeric,
    n: chex.Numeric,
) -> chex.Numeric:
  """Divide standard deviation over a sum of iid terms.

  Consider K random variables `X_1,  ..., X_K` iid with standard deviation
  `mu`.
  `std(X_1 + ... X_K) = sqrt(K) mu`, thus `mu` per step should satisfy:
  `mu = std(X_1 + ... X_K) / sqrt(K)`.

  Args:
    target_std: target standard deviation of the sum.
    n: number of summands in the sum.
  Returns:
    standard deviation per step
  """
  return target_std / jnp.sqrt(n)


def divide_std_over_avg(
    target_std: chex.Numeric,
    n: chex.Numeric,
) -> chex.Numeric:
  """Divide standard deviation over an average of iid terms.

  Consider K random variables `X_1,  ..., X_K` iid with standard deviation
  `mu`.
  `std((X_1 + ... X_K)/K) = mu / sqrt(K)`, thus `mu` per step should satisfy:
  `mu = std((X_1 + ... X_K)/K) * sqrt(K)`.

  Args:
    target_std: target standard deviation of the average.
    n: number of summands in the average.
  Returns:
    standard deviation per step
  """
  return target_std * jnp.sqrt(n)
