#!/bin/bash
# coding=utf-8
# Copyright 2019 The Authors of RL Reliability Metrics.
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

# Test nightly release: ./test_release.sh nightly
# Test stable release: ./test_release.sh stable

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "test_release [nightly|stable]"
  exit 1
fi

run_tests() {
  echo "run_tests $1 $2"

  # Install necessary python version
  pyenv install --list
  pyenv install -s $1
  pyenv global $1

  TMP=$(mktemp -d)
  # Create and activate a virtualenv to specify python version and test in
  # isolated environment. Note that we don't actually have to cd'ed into a
  # virtualenv directory to use it; we just need to source bin/activate into the
  # current shell.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate


  # TensorFlow isn't a regular dependency because there are many different pip
  # packages a user might have installed.
  if [[ $2 == "nightly" ]] ; then
    pip install tf-nightly==1.15.0.dev20190821

    # Run the tests
    python setup.py test

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/
  elif [[ $2 == "stable" ]] ; then
    pip install tensorflow

    # Run the tests
    python setup.py test --release

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/ --release
  elif [[ $2 == "preview" ]] ; then
    pip install tf-nightly-2.0-preview

    # Run the tests
    python setup.py test

    # Install tf_agents package.
    WHEEL_PATH=${TMP}/wheel/$1
    ./pip_pkg.sh ${WHEEL_PATH}/
  else
    echo "Error unknown option only [nightly|stable]"
    exit
  fi

  pip install ${WHEEL_PATH}/rl_reliability_metrics_*.whl

  # Move away from repo directory so "import tf_agents" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import rl_reliability_metrics')

  # Deactivate virtualenv
  deactivate
}

# Test on Python2.7
run_tests "2.7" $1
# Test on Python3.6.1
run_tests "3.6.1" $1

