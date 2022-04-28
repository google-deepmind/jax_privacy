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

"""Privacy accounting."""

from jax_privacy.src.accounting import accountant
from jax_privacy.src.accounting import calibrate
from jax_privacy.src.accounting import std_utils


Accountant = accountant.Accountant
divide_std_over_avg = std_utils.divide_std_over_avg
divide_std_over_sum = std_utils.divide_std_over_sum
calibrate_steps = calibrate.calibrate_steps
compute_epsilon = calibrate.compute_epsilon
calibrate_batch_size = calibrate.calibrate_batch_size
calibrate_noise_multiplier = calibrate.calibrate_noise_multiplier
