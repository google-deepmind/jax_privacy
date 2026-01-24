# coding=utf-8
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

"""Configuration file for the Sphinx documentation builder.

This file contains all the configuration settings for Sphinx to generate
the documentation for the jax_privacy library.
"""

# sphinx-build -b html . _build/html

import os
import sys
from typing import Protocol
from unittest import mock


class MockDataclassInstance(Protocol):
  pass


class MockTypeshed(mock.MagicMock):

  def __getattr__(self, name):
    if name == 'DataclassInstance':
      return MockDataclassInstance
    return super().__getattr__(name)


# Inject this module into sys.modules so Python finds it
sys.modules['_typeshed'] = MockTypeshed()

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'JAX Privacy'
copyright = '2025, Google DeepMind'  # pylint: disable=redefined-builtin
author = 'Google DeepMind'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'myst_nb',
    'sphinx_collections',
    'sphinx.ext.doctest',
    'sphinx_autodoc_typehints',
]

autodoc_type_aliases = {
    'ArrayLike': 'jax.typing.ArrayLike',
    'ArrayTree': 'chex.ArrayTree',
    'PydanticDataclass': 'pydantic.PydanticDataclass',
}

autosummary_generate = True

# Configure autodoc settings
autodoc_typehints = 'signature'
autoclass_content = 'both'
autodoc_member_order = 'bysource'
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

nb_execution_mode = 'off'
suppress_warnings = ['misc.highlighting_failure']

# We have to copy examples to include them in the docs. Without it won't work.
collections = {
    'examples': {
        'driver': 'copy_folder',
        'source': '../examples',  # Path from conf.py to your real examples
        'ignore': 'BUILD',
    }
}
