# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess

#sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../../apps/'))



# call doxygen to generate the XML files
subprocess.call('cd ..; doxygen Doxyfile.in', shell=True)



# -- Project information -----------------------------------------------------

project = 'The leuven Library and Framework Documentation'
copyright = '2020, M.Novak'
author = 'Mihaly Novak'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',    # autogenerate documentation i.e. the .rst files
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',    # to have the [source] option in the doc
    'nbsphinx',               # for jupiter notebook conversions
    'sphinx.ext.napoleon',    # for the google-style doc
    'breathe',                # mixed doxygen and sphinx doc with breathe
    'sphinxcontrib.bibtex'    # to use bibtex
]

bibtex_bibfiles = ['bibfile.bib']

# set path to the doxygen-generated XML for breathe
breathe_projects = { 'The leuven Library and Framework Documentation': '../doxygen/xml' }

##breathe_projects_source = {
##     "auto" : ( "../../", ["matrix_multiplication.c"] )
##     }

numfig = True
source_suffix = '.rst'
master_doc = 'index'

# option for autodoc
autodoc_member_order = 'bysource'  # default is alphabetical

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
  "preamble": r"""
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{bm}
    \usepackage{bbm}
  """,
#  'fncychap': '\\usepackage[Conny]{fncychap}',
#    \usepackage{mathtools}
}


##
## generate `requirements` file for rtfd:
## conda env export -f environment.yml
## then strip down the 'environment.yml' into a 'requirements' file by simple
## listing all packages
