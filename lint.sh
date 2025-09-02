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

# Runs flake8 and pylint on all python files.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/jax-privacy-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install flake8 pytype pylint pylint-exit
pip install -r requirements-dev.txt

# Lint with flake8.
flake8 `find jax_privacy -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.

# Configure pylint-exit to fail the script on error, warning, refactor, 
#and conversion messages.
PYLINT_EXIT_ARGS="-efail -wfail -cfail -rfail"

# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find jax_privacy -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_EXIT_ARGS $?
# Disable `protected-access` warnings and `missing-module-docstring` convention for tests.
pylint --rcfile=.pylintrc `find jax_privacy -name '*_test.py' | xargs` -d W0212,C0114 || pylint-exit $PYLINT_EXIT_ARGS $?

# Check types with pytype.
pytype `find jax_privacy/ -name "*py" | xargs` -k
pytype `find examples/ -name "*py" | xargs` -k


set +u
deactivate
echo "All lint checks passed. Congrats!"
