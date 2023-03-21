# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib import metadata

project = "tuned-lens"
copyright = "2023, FAR AI"
html_title = "Tuned Lens 🔎"
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
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/AlignmentResearch/tuned-lens",
    "source_branch": "main",
    "source_directory": "docs/source",
    "light_css_variables": {
        "sidebar-item-font-size": "85%",
    },
}