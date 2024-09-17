import sys

if sys.platform.startswith("win"):
    import pyMSVC

    environment = pyMSVC.setup_environment()
    print(environment)


from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        ("common_cython.pyx", "large_turns.pyx", "boundary_contact.pyx"),
        compiler_directives={"linetrace": True, "binding": True, "profile": True},
        annotate=True,
    ),
    include_dirs=[np.get_include()],
)
