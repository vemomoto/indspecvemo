'''
Created on 14.02.2017

@author: Samuel
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

NAME = 'settriec'

if os.name == "posix":
    parlinkargs = ['-fopenmp']
    parcompileargs = ['-fopenmp']
else: 
    parlinkargs = ['/openmp']
    parcompileargs = ['/openmp']

extensions = [Extension(NAME, [NAME+'.pyx'],
                        extra_compile_args=['-std=c++11', '-O3']+parcompileargs,
                        extra_link_args=parlinkargs,
                        include_dirs=["datrie\\datrie-master\\src",
                                      "datrie\\datrie-master\\libdatrie",
                                      "datrie\\datrie-master\\libdatrie\datrie"
                                      ],
                        )
              ]

setup(name=NAME,
      ext_modules = cythonize(extensions, language="c++"),
      )
