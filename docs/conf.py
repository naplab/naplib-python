# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/main/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from distutils.version import LooseVersion

import matplotlib

# Use RTD Theme
import sphinx_rtd_theme
import sphinx_gallery

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "naplib"
copyright = "2023, Neural Acoustic Processing Lab, Columbia University, New York, NY"
authors = u"Gavin Mischler, Vinay Raghavan, Menoua Keshishian"

# The short X.Y version
# Find naplib version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "..", "naplib", "__init__.py")):
    if line.startswith("__version__ = "):
        version = line.strip().split()[2][1:-1]

# The full version, including alpha/beta/rc tags
release = "alpha"

# -- Extension configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    'sphinx_gallery.gen_gallery',
]

if LooseVersion(sphinx_gallery.__version__) < LooseVersion('0.2'):
    raise ImportError('Must have at least version 0.2 of sphinx-gallery, got '
                      '%s' % (sphinx_gallery.__version__,))

matplotlib.use('agg')

# -- sphinxcontrib.rawfiles
#rawfiles = ["CNAME"]

# -- numpydoc
# Below is needed to prevent errors
numpydoc_show_class_members = False

# -- sphinx.ext.autosummary
autosummary_generate = True

# -- sphinx.ext.autodoc
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("http://scikit-learn.org/dev", None),
}

# -- sphinx options ----------------------------------------------------------
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
modindex_common_prefix = ["naplib."]

pygments_style = "sphinx"
smartquotes = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    # 'includehidden': False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "logo_only": True,
}
html_logo = "./figures/NAPLabLogo_darkgrey.png"
html_favicon = "./figures/favicon-32x32.ico"

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    "github_user": "naplab",
    "github_repo": "naplib-python",
    "github_version": "main/docs/",
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "naplibdoc"

# -- Options for LaTeX output ------------------------------------------------


def setup(app):
    # to hide/show the prompt in code examples:
    app.add_js_file("js/copybutton.js")

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "naplib.tex", "naplib Documentation", authors, "manual")
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "naplib", "naplib Documentation", [authors], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "naplib",
        "naplib Documentation",
        authors,
        "naplib",
        "One line description of project.",
        "Miscellaneous",
    )
]

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}

sphinx_gallery_conf = {
    'doc_module': 'naplib',
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'naplib': None,
    },
    'ignore_pattern': r'noinclude\.py',
    ## Add ability to link to mini-gallery under each function/class
    # directory where function/class granular galleries are stored
    'backreferences_dir'  : 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and naplib in a tuple of strings.
    'doc_module'          : ('sphinx_gallery', 'naplib'),
}


