# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

# Runs unit tests locally.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/jax-privacy-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Build the package.
pip install .

# Install dev dependencies.
pip install -r requirements-dev.txt

# Change directory to avoid importing the package from repo root.
# Runing tests from the root can cause an ImportError for cython
# (.pyx) modules and possibly other issues.
mkdir _testing && cd _testing

# Main tests.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy \
  -k "not dp_updater_test and not evaluator_test and not matrix_factorization and not grad_clipping_sharded_test and not distributed_noise_generation_test"

# The matrix_factorization tests are expensive, and require the correct
# HYPOTHESIS_PROFILE to limit the number of examples tested.
export HYPOTHESIS_PROFILE=dpftrl_default
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "matrix_factorization"

# Isolate tests that use `chex.set_n_cpu_device()`.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "dp_updater_test"
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "evaluator_test"
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "grad_clipping_sharded_test"
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "distributed_noise_generation_test"

cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
