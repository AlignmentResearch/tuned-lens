# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib import metadata

project = "Tuned Lens"
copyright = ("2023", "FAR AI")
author = (
    "Nora Belrose"
    " Zach Furman,"
    " Logan Smith,"
    " Danny Halawi,"
    " Lev McKinney,"
    " Igor Ostrovsky,"
    " Stella Biderman,"
    " Jacob Steinhardt"
)
release = metadata.version("tuned_lens")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
]

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
