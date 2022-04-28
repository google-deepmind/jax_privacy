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

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/jax-privacy-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pytype pylint pylint-exit
pip install -r requirements.txt

# Lint with flake8.
flake8 `find jax_privacy -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find jax_privacy -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find jax_privacy -name '*_test.py' | xargs` -d W0212 || pylint-exit $PYLINT_ARGS $?

# Build the package.
python setup.py install

# Check types with pytype.
pytype `find jax_privacy/src/ -name "*py" | xargs` -k

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
pip install -r requirements.txt
mkdir _testing && cd _testing

# Main tests.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "not updater_test"

# Isolate tests that use `chex.set_n_cpu_device()`.
pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs jax_privacy -k "updater_test"
cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
