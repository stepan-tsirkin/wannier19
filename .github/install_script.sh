#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>
# Copied from Z2Pack https://github.com/Z2PackDev/Z2Pack

# Be verbose, and stop with error as soon there's one
set -ev

pip install codecov
pip install -U pip setuptools wheel

# install optional dependencies
pip install tbmodels pythtb spglib

case "$INSTALL_TYPE" in
    dev)
        pip install -e .
        ;;
    dev_sdist)
    #     python setup.py sdist
    #     ls -1 dist/ | xargs -I % pip install dist/%[dev]
    #     ;;
    # dev_bdist_wheel)
    #     python setup.py bdist_wheel
    #     ls -1 dist/ | xargs -I % pip install dist/%[dev]
    #     ;;
esac
