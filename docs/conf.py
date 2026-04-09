import subprocess, os

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "SDSL"
author  = "TAU-CGL"
release = "1.0"
copyright = "2024, TAU-CGL"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "breathe",          # bridges Doxygen XML → Sphinx
    "exhale",           # auto-generates C++ API pages from Breathe
    "myst_parser",      # lets us write tutorials in Markdown
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # NumPy/Google docstring style for Python pages
]

# MyST — enable useful extensions (math, colon fences, etc.)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# ---------------------------------------------------------------------------
# Breathe
# ---------------------------------------------------------------------------
breathe_projects        = {"sdsl": "./doxygen/xml"}
breathe_default_project = "sdsl"

# ---------------------------------------------------------------------------
# Exhale — auto-generates api/ from Breathe's parsed Doxygen tree
# ---------------------------------------------------------------------------
exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "C++ API Reference",
    "doxygenStripFromPath":  "../sdsl/include",
    "createTreeView":        True,
    # Use a Doxygen call so you can build docs with: sphinx-build docs/ docs/_build
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    open(os.path.join(os.path.dirname(__file__), "Doxyfile")).read(),
}

# Primary domain for autodoc directives in .rst files
primary_domain = "cpp"

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "SDSL Docs"

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path   = ["_templates"]
exclude_patterns = ["_build", "doxygen"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}
