.. Copyright 2026 DeepMind Technologies Limited.
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

#############
Core Library
#############

.. currentmodule:: jax_privacy


Public API
----------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:

  batch_selection
  clipping
  noise_addition
  auditing

Matrix Factorization
--------------------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:
  :template: autosummary/module.rst

  matrix_factorization
  matrix_factorization.banded
  matrix_factorization.buffered_toeplitz
  matrix_factorization.checks
  matrix_factorization.dense
  matrix_factorization.optimization
  matrix_factorization.sensitivity
  matrix_factorization.streaming_matrix
  matrix_factorization.toeplitz

Experimental Modules
--------------------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:

  experimental.execution_plan
  experimental.compilation_utils
  experimental.accounting
  experimental.monte_carlo
