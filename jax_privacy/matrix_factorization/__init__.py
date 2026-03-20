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

"""Public API for matrix factorization."""

from jax_privacy.matrix_factorization import banded as banded
from jax_privacy.matrix_factorization import buffered_toeplitz as buffered_toeplitz
from jax_privacy.matrix_factorization import checks as checks
from jax_privacy.matrix_factorization import dense as dense
from jax_privacy.matrix_factorization import optimization as optimization
from jax_privacy.matrix_factorization import sensitivity as sensitivity
from jax_privacy.matrix_factorization import streaming_matrix as streaming_matrix
from jax_privacy.matrix_factorization import toeplitz as toeplitz
from jax_privacy.matrix_factorization.streaming_matrix import StreamingMatrix

__all__ = [
    'banded',
    'buffered_toeplitz',
    'checks',
    'dense',
    'optimization',
    'sensitivity',
    'streaming_matrix',
    'toeplitz',
    'StreamingMatrix',
]
