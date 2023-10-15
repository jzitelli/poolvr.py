#!/bin/env python
from setuptools import setup, Extension
from codecs import open
from os import path, listdir
#from Cython.Build import cythonize
from sys import exec_prefix

here = path.dirname(path.abspath(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='poolvr',
    version='0.0.1',
    description='Python VR pool simulator',
    packages=['poolvr'],
    long_description=long_description,
    url='https://github.com/jzitelli/poolvr.py',
    author='Jeffrey Zitelli',
    author_email='jeffrey.zitelli@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='pool billiards event based physics',
    install_requires=['numpy', 'pillow'],
    extras_require={},
    # package_data={
    # },
    data_files=[],
)
