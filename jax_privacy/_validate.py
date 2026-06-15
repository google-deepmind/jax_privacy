# Copyright 2026 DeepMind Technologies Limited.
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

"""Centralized validation utilities for jax_privacy."""

import jax
import numpy as np

# ---------------------------------------------------------------------------
# Generic validators
# ---------------------------------------------------------------------------


def non_negative(**kwargs):
  """Validates that all values are non-negative."""
  for name, value in kwargs.items():
    if value < 0:
      raise ValueError(f'Expected {name}={value} >= 0.')


def positive(**kwargs):
  """Validates that all values are positive."""
  for name, value in kwargs.items():
    if value <= 0:
      raise ValueError(f'Expected {name}={value} > 0.')


def in_range(lo, hi, **kwargs):
  """Validates that all values are in [lo, hi]."""
  for name, value in kwargs.items():
    if not lo <= value <= hi:
      raise ValueError(f'Expected {name}={value} in [{lo}, {hi}].')


def batch(pytree) -> int:
  """Validates a batch pytree and returns the batch size.

  Checks that the pytree has at least one leaf, all leaves are arrays with
  at least one dimension, and all leaves have the same size along axis 0.

  Args:
    pytree: A pytree of arrays representing a batch of data.

  Returns:
    The batch size (size of the first dimension).

  Raises:
    ValueError: If the pytree is empty, contains non-array leaves, leaves
      with zero dimensions, or leaves with inconsistent first-axis sizes.
  """
  leaves = jax.tree.leaves(pytree)
  if not leaves:
    raise ValueError('Batch pytree is empty or contains no leaves.')
  for i, leaf in enumerate(leaves):
    if not hasattr(leaf, 'shape'):
      raise ValueError(
          f'Expected all batch leaves to be arrays, but leaf {i} has type '
          f'{type(leaf).__name__}.'
      )
    if leaf.ndim == 0:
      raise ValueError(
          'Expected all batch leaves to have at least one dimension, but '
          f'leaf {i} is a scalar.'
      )
  sizes = {leaf.shape[0] for leaf in leaves}
  if len(sizes) != 1:
    raise ValueError(
        'All batch leaves must have the same size along axis 0, '
        f'got sizes: {sorted(sizes)}.'
    )
  return sizes.pop()


def strategy(value, max_size):
  """Validates BandMF strategy coefficients.

  Checks that the strategy is a 1D array with size in ``[1, max_size]``.

  Args:
    value: The strategy coefficients as an array-like.
    max_size: The maximum allowed size (typically the number of iterations).

  Raises:
    ValueError: If the strategy is not 1D or its size is out of range.
  """
  arr = np.asarray(value)
  if arr.ndim != 1:
    raise ValueError(f'strategy must be a 1D array, got {arr.ndim}D.')
  if arr.size == 0 or arr.size > max_size:
    raise ValueError(
        f'strategy size must be in [1, {max_size}], got {arr.size}.'
    )


def multi_owner(example_ids, user_ids):
  """Validates parallel edge-list arrays for a MultiOwnerGraph."""
  example_ids = np.asarray(example_ids)
  user_ids = np.asarray(user_ids)
  if example_ids.ndim != 1 or user_ids.ndim != 1:
    raise ValueError('example_ids and user_ids must be 1D arrays.')
  if len(example_ids) != len(user_ids):
    raise ValueError('example_ids and user_ids must have the same length')
  if len(example_ids) == 0:
    raise ValueError('empty graphs are not allowed.')
  pairs = np.stack([example_ids, user_ids], axis=1)
  if len(np.unique(pairs, axis=0)) < len(pairs):
    raise ValueError('Duplicate (example, user) id pairs are not allowed.')
