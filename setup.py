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

"""Setup for pip package."""

import unittest
import setuptools


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


def test_suite():
  test_loader = unittest.TestLoader()
  all_tests = test_loader.discover('jax_privacy', pattern='*_test.py')
  return all_tests


setuptools.setup(
    name='jax_privacy',
    version='1.0.0',
    description='Algorithms for Privacy-Preserving Machine Learning in JAX.',
    url='https://github.com/google-deepmind/jax_privacy',
    author='Google DeepMind',
    author_email='jax-privacy-dev@google.com',
    # Contained modules and scripts.
    packages=setuptools.find_namespace_packages(
        exclude=['*_test.py', 'examples*', 'experiments*']
    ),
    install_requires=_parse_requirements('requirements.txt'),
    requires_python='>=3.11',
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.test_suite',
    include_package_data=True,
    zip_safe=False,
)
