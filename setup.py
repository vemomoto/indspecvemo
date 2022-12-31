from os import path
import sys

import Cython.Compiler.Options
from setuptools import Extension, setup

from os import path

# ==== Basic settings ====

# Name of the extension
PACKAGE_NAME = "indspecvemo"
EXT_NAME = "settrie"


Cython.Compiler.Options.annotate = True

# ==== Create extension objects summarizing the information needed for the build ====

extensions = [Extension(EXT_NAME, [path.join(PACKAGE_NAME, "settrie", EXT_NAME+'.pyx')],
                        language="c++", 
                        extra_compile_args=['-std=c++11', '-O3', '/openmp'],
                        extra_link_args=['/openmp'],
                        include_dirs=[path.join(PACKAGE_NAME, "settrie"),
                                      path.join(PACKAGE_NAME, "settrie", "datrie-master", "src"),
                                      path.join(PACKAGE_NAME, "settrie", "datrie-master", "libdatrie"),
                                      path.join(PACKAGE_NAME, "settrie", "datrie-master", "libdatrie", "datrie"),
                                      ],
                        )
              ]

if __name__ == "__main__":
    print("Using the following executable:", sys.executable)

    setup(
        name=PACKAGE_NAME,
        version=0.1,
        ext_modules=extensions, 
        setup_requires=['cython'],
        packages=[PACKAGE_NAME],
        install_requires=[
            'numpy', 
            'scipy', 
            'matplotlib', 
            'pandas', 
            'cython',
            'datrie',
            'pebble',
            'numdifftools', 
            'autograd',
            'hybrid_vector_model',
            'vemomoto_core',
            'ci_rvm',
            ], 
        python_requires='>=3.8',
        zip_safe=False,
        # metadata to display on PyPI
        license="LGPLv3",
        author="Samuel Fischer",
    )
