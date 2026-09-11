"""Configure the Sphinx documentation builder."""

project = "jmstate"
release = ""
version = ""
copyright = "2026, Félix Laplante"
author = "Félix Laplante"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
autosummary_generate = True
add_module_names = False
napoleon_use_ivar = True
napoleon_attr_annotations = True
suppress_warnings = ["docutils", "ref.ref", "ref.term"]
nb_execution_mode = "off"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "jmstate"
html_logo = "_static/jmstate-logo.svg"
html_favicon = "_static/jmstate-logo.svg"
html_theme_options = {
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/felixlaplante0/jmstate",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"**": []}
