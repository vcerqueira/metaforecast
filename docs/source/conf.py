# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))

project = 'metaforecast'
copyright = '2024, Vitor Cerqueira'
author = 'Vitor Cerqueira'
release = '0.1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    'sphinx.ext.viewcode',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx_link',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_mock_imports = [
    'numpy',
    'scipy',
    'pandas',
    'tqdm',
    'cython',
    'torch',
    'sklearn',
    'datasetsforecast',
    'statsforecast',
    'mlforecast',
    'neuralforecast',
    'numba',
    'arch',
    'lightgbm',
    'tslearn',
    'tslearn.barycenters',
    'scipy.interpolate',
    'scipy.stats',
    'statsmodels',
    'statsmodels.tsa',
    'statsmodels.tsa.api',
    'statsmodels.compat',
    'statsmodels.tools',
    'packaging',
    'patsy'
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['docs/source/_static']

autosummary_generate = True
add_module_names = False
