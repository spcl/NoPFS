# -*- coding: utf-8 -*-
import os
import platform
import subprocess
from setuptools import setup
from distutils.command.install import INSTALL_SCHEMES

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

LIB_EXTENSION = "so"
if platform.system() == "Darwin":
    LIB_EXTENSION = "dylib"

with open("MANIFEST", "w") as manifest:
    manifest.write("libhdmlp/build/libhdmlp." + LIB_EXTENSION)

try:
    out = subprocess.check_output(['cmake', '--version'])
except OSError:
    raise RuntimeError("CMake must be installed")
extdir = os.path.abspath("libhdmlp/")
builddir = os.path.abspath("libhdmlp/build/")
if not os.path.exists(builddir):
    os.mkdir(builddir)
cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + builddir, '-DBUILD_SHARED_LIBS=ON']
build_args = ['--', '-j4']
subprocess.check_call(['cmake', extdir] + cmake_args, cwd=extdir)
subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=extdir)


with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='hdmlp',
    version='0.1',
    author='Roman Böhringer',
    description='Hierarchical Distributed Machine Learning Prefetcher',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['hdmlp', 'hdmlp.lib.torch', 'hdmlp.lib.transforms'],
    data_files=[('', ['libhdmlp/build/libhdmlp.' + LIB_EXTENSION])],
    zip_safe=False,
    install_requires=[
        'pillow'
    ]
)
