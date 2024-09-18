"""
This setup.py script configures and builds Cython extensions for the project.

The following Cython modules are built:
- src.utils.common_cython: Utility functions shared across the project.
- src.analysis.large_turns: Functions for detecting and analyzing large turns in trajectories.
- src.analysis.boundary_contact: Functions for boundary contact analysis.

The build process:
- Uses NumPy headers for compilation.
- Enables line tracing, binding, and profiling features in Cython for debugging.
- Generates .so files in place, next to their respective .pyx source files.

To build the Cython modules, run:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define Cython extensions for various modules
extensions = [
    Extension(
        "src.utils.common_cython",
        sources=["src/utils/common_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "src.analysis.large_turns",
        sources=["src/analysis/large_turns.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "src.analysis.boundary_contact",
        sources=["src/analysis/boundary_contact.pyx"],
        include_dirs=[np.get_include()],
    ),
]

# Setup function to build the Cython extensions
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"linetrace": True, "binding": True, "profile": True},
        annotate=True,  # Generates HTML file for annotated Cython code
    ),
    include_dirs=[np.get_include()],  # Include NumPy headers
    options={
        "build_ext": {
            "inplace": True,  # Output .so files next to .pyx source files
        },
    },
)
