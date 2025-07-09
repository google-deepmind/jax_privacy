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

r"""Test utilities, including configuring hypothesis profiles.
"""

import os
import sys
import hypothesis


def configure_hypothesis():
  """Loads the correct hypothesis profile."""
  # We could simply put this code at the module level, and it would run
  # automatically in whatever tests import test_utils. However, this makes
  # it hard to discover how hypothesis is configured, and if no other
  # functionality from this file is used, an unused import lint warning
  # would need to be suppressed. So we instead expose this as a function
  # that tests can explicitly call.

  hypothesis.settings.register_profile(
      'dpftrl_rigorous',
      deadline=None,  # 200ms disable
      derandomize=False,
      max_examples=100,
      # Change to "verbose" to print out each test example selected
      verbosity=hypothesis.Verbosity.normal,
  )

  hypothesis.settings.register_profile(
      'dpftrl_default',
      database=None,
      deadline=None,  # 200ms disable
      derandomize=True,
      max_examples=10,
      verbosity=hypothesis.Verbosity.normal,
  )
  profile = (
      'dpftrl_default' if os.getenv('UNITTEST_ON_FORGE') else 'dpftrl_rigorous'
  )
  profile = os.getenv('HYPOTHESIS_PROFILE', default=profile)
  hypothesis.settings.load_profile(profile)
  # If the wrong profile is used, it can cause tests to timeout, so make
  # sure we print this so it can easily be found in logs:
  print(
      f'Using hypothesis profile "{profile}":\n{hypothesis.settings.default}',
      file=sys.stderr,
      flush=True,
  )


def scale_max_examples(scale: float) -> int:
  """Scales the current max_examples.

  Usually used via
  `@hypothesis.settings(max_examples=test_utils.scale_max_examples(0.3))`
  to scale down the number of examples for expensive tests.

  Args:
    scale: The scale factor to apply to the default max_examples.

  Returns:
    The scaled max_examples, at least 1.
  """
  return max(1, int(round(scale * hypothesis.settings.default.max_examples)))
