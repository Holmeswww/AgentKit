# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information

project = 'AgentKit'
copyright = '2024, AgentKit Developers'
author = 'Yue Wu'

release = '0.1'
version = '0.1.8.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.egg-info", "dist"]

# -- Options for HTML output

html_favicon = 'img/favicon.ico'

html_theme = 'sphinx_rtd_theme'

html_logo = "img/AgentKit.png"

html_theme_options = {
    "logo_only": True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Napoleon settings
napoleon_google_docstring = True
napoleon_use_param = False
