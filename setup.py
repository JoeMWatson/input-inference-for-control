from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from distutils.errors import CompileError
from warnings import warn
import os
import sys
from glob import glob

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # name='input-inference-for-control',
    name='pi2c',
    version='0.0.0',
    install_requires=requirements,
    description="Input Inference for Stochastic Optimal Control",
    author="",
    author_email='',
    license="",
    url='',
    # packages=['input-inference-for-control'],
    setup_requires=['future'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
    keywords=['optimal control', 'inference', 'reinforcement learning'],
    platforms="ALL",
)
