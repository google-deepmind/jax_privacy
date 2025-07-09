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

"""Public API for the noise_addition module."""
# pylint: disable=g-importing-member

from .additive_privatizers import gaussian_privatizer
from .additive_privatizers import matrix_factorization_privatizer
from .distributed_noise_generation import infer_state_sharding
from .distributed_noise_generation import streaming_matrix_to_sharded_privatizer
from .distributed_noise_generation import streaming_matrix_to_single_machine_privatizer
from .gradient_privatizer import GradientPrivatizer
