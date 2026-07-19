"""Sphinx configuration for the PMRIPrep documentation site."""

from importlib.metadata import version as _get_version

# -- Project information -------------------------------------------------------
project = "PMRIPrep"
author = "PMRIPrep Developers"
copyright = "2026, PMRIPrep Developers"  # noqa: A001

# Read the version from the installed package (run ``pip install -e .[doc]`` so
# this reflects the current pyproject version rather than stale editable metadata).
release = _get_version("pmriprep")
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinxcontrib.apidoc",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- napoleon (NumPy-style docstrings) -----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = True
# Render class ``Attributes`` (e.g. dataclass fields) as inline ``:ivar:`` entries
# rather than separate ``.. attribute::`` directives, which would otherwise duplicate
# the same fields that autodoc documents via ``:members:``.
napoleon_use_ivar = True
# Render the workflow ``Inputs``/``Outputs`` docstring sections as parameter tables.
napoleon_custom_sections = [
    ("Inputs", "params_style"),
    ("Outputs", "returns_style"),
]

# -- autodoc -------------------------------------------------------------------
# Every runtime dependency imports cleanly in the docs environment, so nothing is
# mocked; the package is pip-installed (``.[doc]``) and imported directly.
autodoc_mock_imports = []
# ``ignore-module-all`` documents each object only in the module that defines it;
# without it, names re-exported through package ``__init__`` ``__all__`` lists are
# documented at both the package and the defining module (a duplicate object).
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
# Generated API pages index several public objects with the same short name.
# Suppress only Python-domain ambiguity warnings; all other warnings remain fatal.
suppress_warnings = ["ref.python"]

# -- sphinxcontrib-apidoc (auto-generates API stubs into docs/api/ at build) ---
apidoc_module_dir = "../pmriprep"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--module-first", "-T"]
apidoc_excluded_paths = ["*/tests/*", "*conftest.py"]

# -- sphinxcontrib-bibtex ------------------------------------------------------
bibtex_bibfiles = ["../pmriprep/data/boilerplate.bib"]
bibtex_reference_style = "author_year"

# -- intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nipype": ("https://nipype.readthedocs.io/en/latest/", None),
    "niworkflows": ("https://www.nipreps.org/niworkflows/", None),
    "bids": ("https://bids-standard.github.io/pybids/", None),
}
