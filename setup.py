#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import print_function

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname),'r',encoding='utf-8').read()

setup(
    name = "DeepFrog",
    version = "0.0.1", #also edit in __init__.py
    author = "Maarten van Gompel",
    author_email = "proycon@anaproy.nl",
    description = (""),
    license = "GPL",
    keywords = "nlp computational_linguistics",
    url = "https://github.com/proycon/deepfrog",
    packages=['deepfrog'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    zip_safe=False,
    include_package_data=True,
    install_requires=[ 'torch', 'numpy', 'seqeval','tqdm','tensorboardX' ],
    entry_points = {   'console_scripts': [ 'deepfrog-tagger = deepfrog.tagger:main' ] }
)
