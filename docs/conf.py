# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('./..'))


# -- Project information -----------------------------------------------------

project = 'crosslingual-information-retrieval'
copyright = '2021, Min Duc Bui, Jakob Langenbahn, Niklas Sabel'
author = 'Min Duc Bui, Jakob Langenbahn, Niklas Sabel'

release = '0.1'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme'
]

templates_path = []

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = []
