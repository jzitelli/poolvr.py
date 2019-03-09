#!/bin/env python
from setuptools import setup
from codecs import open
from os import path, listdir
from Cython.Build import cythonize
from distutils.extension import Extension
import sys
here = path.dirname(path.abspath(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='poolvr.py',
    version='0.0.1',
    packages=['poolvr'],
    ext_modules=cythonize([Extension('poolvr.physics.events', [path.join('poolvr', 'physics', 'events.pyx')],
                                     include_dirs=[path.join(sys.exec_prefix, 'lib', 'site-packages', 'numpy', 'core', 'include'), '.'],
                                     extra_compile_args=["-Zi", "/O2"],
                                     extra_link_args=["-debug"]),
                           Extension('poolvr.physics.pool_physics', [path.join('poolvr', 'physics', 'pool_physics.pyx')],
                                     include_dirs=[path.join(sys.exec_prefix, 'lib', 'site-packages', 'numpy', 'core', 'include'), '.'],
                                     extra_compile_args=["-Zi", "/O2"],
                                     extra_link_args=["-debug"]),
                           Extension('poolvr.app', [path.join('poolvr', 'app.pyx')],
                                     extra_compile_args=["-Zi", "/O2"],
                                     extra_link_args=["-debug"])]),
    description='Python VR pool simulator',
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
    package_data={
        'poolvr': [path.join('shaders', filename) for filename in listdir(path.join('poolvr', 'shaders'))],
    },
    data_files=[],
    scripts=[path.join('scripts', 'gen_assets.py')],
    entry_points={
        'console_scripts': [
            'poolvr = poolvr.__main__:main'
        ]
    }
)
