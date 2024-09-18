from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

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

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"linetrace": True, "binding": True, "profile": True},
        annotate=True,  # This will generate HTML files for annotated code
    ),
    include_dirs=[np.get_include()],
    options={
        "build_ext": {
            "inplace": True,  # Place .so files next to the .pyx files
        },
    },
)
