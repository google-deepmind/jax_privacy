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

"""Library for contribution matrix constructors.

A contrib matrix (aka, participation matrix) contains columns of length `n`
(num_iterations), with each column corresponding to a possible pariticpation
pattern. A +1 or -1 entry corresponds to an iteration where the user
participated, with a 0 corresponding
to non-participation.
"""

import jax.numpy as jnp
import numpy as np


def _plus_minus_matrix(num_rows):
  """Creates a +1/-1 matrix with num_rows, first row all 1s.

  Args:
    num_rows: The number of rows in the returned matrix.

  Returns:
    A (num_rows x 2**(num_rows - 1)) shaped matrix where the columns
    include all possible length num_rows vectors with +1/-1 entries where the
    first entry is always +1.
  """
  n = num_rows - 1
  mask = np.array(
      [
          [x == '1' for x in np.binary_repr(k, width=num_rows)]
          for k in range(2**n)
      ]
  ).T
  m = np.ones(shape=mask.shape)
  m[mask] = -1
  return m


def epoch_participation_matrix(n: int, num_epochs: int) -> np.ndarray:
  """Constructs participation matrix for epoch-based training.

  Args:
    n: Number of iterations (rows in the returned matrix).
    num_epochs: Number of epochs, must divide n.

  Returns:
    A participation matrix. The number of columns will in general be exponential
    in n and num_epochs.
  """
  assert n % num_epochs == 0
  steps_per_epoch = n // num_epochs
  num_pm_variants = 2 ** (num_epochs - 1)
  num_constraints = num_pm_variants * steps_per_epoch
  m = np.zeros(shape=(n, num_constraints))
  row_indexes = np.array(
      [steps_per_epoch * i for i in range(num_epochs)], dtype=np.int32
  )
  pm_block = _plus_minus_matrix(num_rows=num_epochs)
  for offset in range(steps_per_epoch):
    c = offset * num_pm_variants
    m[row_indexes + offset, c : (c + num_pm_variants)] = pm_block
  return m


def epoch_participation_matrix_all_positive(
    n: int, num_epochs: int
) -> np.ndarray:
  """Returns an epoch participation matrix with only +1 entries.

  Suitable for use when X is known to be non-negative, so
  max_u u' X u where u is a column of the participation matrix is always
  achieved by a column with only +1 entries.

  Args:
    n: Number of iterations (rows in the returned matrix).
    num_epochs: Number of epochs, must divide n.

  Returns:
    The participation matrix.
  """
  assert n % num_epochs == 0
  steps_per_epoch = n // num_epochs
  num_constraints = steps_per_epoch
  m = np.zeros(shape=(n, num_constraints))
  row_indexes = np.array(
      [steps_per_epoch * i for i in range(num_epochs)], dtype=np.int32
  )
  for i in range(steps_per_epoch):
    m[row_indexes + i, i] = 1
  return m


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name
def max_min_sensitivity_squared_for_x(
    X: jnp.ndarray, contrib_matrix: np.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Returns sensitivity**2 for X based on constraints in `lt`.

  Args:
    X: The active x-matrix (may or may not be feasible).
    contrib_matrix: The contribution matrix.

  Returns:
    A tuple (min_sensivity^2, max_sensitivity^2) where the min and max are
    over all the participation vectors `u` (columns of `contrib_matrix`).
  """
  s = jnp.diag(contrib_matrix.T @ X @ contrib_matrix)
  max_s, min_s = jnp.max(s), jnp.min(s)
  assert min_s >= -1e-10, min_s
  return max_s, min_s


# pylint:enable=invalid-name
