#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys
import subprocess

with open('README.md') as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='mdr',
    version='0.0.1',
    description='Multi-hop dense retrieval for complex open-domain question answering',
    long_description='text/markdown',
    # license=license,
    python_requires='>=3.6',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)
