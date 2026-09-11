"""Build the jmstate survival-bucket C++ extension."""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"
COMPILE_ARGS = ["/O2"] if WINDOWS else ["-O3", "-pthread"]
LINK_ARGS = [] if WINDOWS else ["-pthread"]

cxxflags = os.environ.get("CXXFLAGS", "")
if not {"-std=c++17", "-std=gnu++17"}.intersection(cxxflags.split()):
    os.environ["CXXFLAGS"] = f"{cxxflags} -std=c++17".strip()

EXTENSION = Pybind11Extension(
    "jmstate.utils._surv_ext",
    ["jmstate/utils/_surv_ext.cpp"],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
    cxx_std=17,
)

setup(ext_modules=[EXTENSION])
