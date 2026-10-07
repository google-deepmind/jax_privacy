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

from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import _validate


class PositiveTest(parameterized.TestCase):

  @parameterized.parameters(1, 0.5, 1e-12)
  def test_accepts_finite_positive(self, value):
    _validate.positive(x=value)

  @parameterized.parameters(0, -1.0, float('nan'), float('inf'))
  def test_rejects_non_positive_or_non_finite(self, value):
    with self.assertRaisesRegex(ValueError, r'Expected x=.* > 0'):
      _validate.positive(x=value)


if __name__ == '__main__':
  absltest.main()
