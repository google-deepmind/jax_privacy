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

"""Algorithms for Privacy-Preserving Machine Learning in JAX."""
from jax_privacy import accounting
from jax_privacy import auditing
from jax_privacy import dp_sgd
from jax_privacy import keras
from jax_privacy import matrix_factorization
from jax_privacy import noise_addition
from jax_privacy import training

# pylint: disable=g-importing-member
# Carefully selected member imports for the top-level public API.
from jax_privacy.experimental import clipped_grad

__version__ = '1.0.0'
