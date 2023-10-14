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
    name='poolvr.py',
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
    keywords='virtual reality vr pool billiards pyopenvr openvr event based physics cue table pocket bumper rail',
    install_requires=['openvr', 'numpy', 'pyopengl', 'pillow'], # 'cyglfw3'],
    extras_require={},
    # package_data={
    # },
    data_files=[],
    scripts=[path.join('scripts', 'gen_assets.py')],
    entry_points={
        'console_scripts': [
            'poolvr = poolvr.__main__:main'
        ]
    },
    # ext_modules=cythonize([Extension('poolvr.physics.coll', [path.join('poolvr', 'physics', 'coll.pyx')],
    #                                  include_dirs=[path.join(exec_prefix, 'lib', 'site-packages', 'numpy', 'core', 'include'), '.'],
    #                                  libraries=['collisions'],
    #                                  library_dirs=[path.join(here, 'poolvr', 'physics')])],
    #                                  # extra_compile_args=["-Zi", "/O2"],
    #                                  # extra_link_args=["-debug"])],
    #                        language_level=3)
)
