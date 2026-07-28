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

"""Experimental JAX Privacy APIs and Implementations.

Submodules:
  discrete_gaussian: Host-side discrete Gaussian sampling for hardened DP-SGD.
  monte_carlo: Monte Carlo privacy accounting.
  compilation_utils: Ahead-of-time compilation helpers for DP training.
  training: Experimental ``DPTrainer`` loop utilities.
"""

from . import discrete_gaussian
from . import monte_carlo
