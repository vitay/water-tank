from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy 

setup(
    ext_modules = cythonize([
        Extension("water_tank.projections.sparse.Sparse",
            ["src/water_tank/projections/sparse/Sparse.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++"),
    ], 
    language_level=3)
)
