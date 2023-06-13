# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Autoscoper"
copyright = f"{date.today().year}, Brown University and Kitware, Inc"
author = "Autoscoper Community"
release = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "notfound.extension",  # Show a better 404 page when an invalid address is entered
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinxcontrib.autoprogram",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["refs.bib"]
myst_enable_extensions = [
    "colon_fence",  # Allow code fence using ::: (see https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html#syntax-colon-fence)
    "html_image",  # Allow using isolated img tags (i.e. not wrapped in any other HTML) to specify attributes like width.
    "linkify",  # Allow automatic creation of links from URLs (it is sufficient to write https://google.com instead of <https://google.com>)
]

myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/SlicerAutoscoperM_Logo_Horizontal.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]
