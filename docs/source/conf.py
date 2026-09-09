# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "metaforecast"
copyright = "2024–2026, Vitor Cerqueira"
author = "Vitor Cerqueira"
release = "0.2.3"

# -- General configuration ---------------------------------------------------

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
    "sphinx.ext.viewcode",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx_link",
    "sphinx_rtd_theme",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "scipy.interpolate",
    "scipy.stats",
    "scipy.special",
    "pandas",
    "tqdm",
    "cython",
    "torch",
    "torch.nn",
    "torch.optim",
    "sklearn",
    "sklearn.gaussian_process",
    "sklearn.gaussian_process.kernels",
    "sklearn.multioutput",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "datasetsforecast",
    "datasetsforecast.evaluation",
    "datasetsforecast.losses",
    "utilsforecast",
    "utilsforecast.evaluation",
    "utilsforecast.losses",
    "statsforecast",
    "mlforecast",
    "mlforecast.target_transforms",
    "neuralforecast",
    "neuralforecast.core",
    "neuralforecast.losses",
    "neuralforecast.losses.numpy",
    "neuralforecast.losses.pytorch",
    "coreforecast",
    "coreforecast.grouped_array",
    "coreforecast.scalers",
    "utilsforecast.compat",
    "utilsforecast.processing",
    "utilsforecast.validation",
    "numba",
    "arch",
    "arch.bootstrap",
    "lightgbm",
    "pytorch_lightning",
    "pytorch_lightning.callbacks",
    "pytorch_lightning.utilities",
    "lightning_fabric",
    "lightning_utilities",
    "importlib_metadata",
    "tslearn",
    "tslearn.barycenters",
    "tslearn.barycenters.dba",
    "statsmodels",
    "statsmodels.tsa",
    "statsmodels.tsa.api",
    "statsmodels.tsa.seasonal",
    "statsmodels.compat",
    "statsmodels.tools",
    "statsmodels.stats",
    "statsmodels.stats.diagnostic",
    "statsmodels.stats.stattools",
    "statsforecast.models",
    "tsfeatures",
    "catboost",
    "joblib",
    "packaging",
    "patsy",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autosummary_generate = False
add_module_names = False

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
