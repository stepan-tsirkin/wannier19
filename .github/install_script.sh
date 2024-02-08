#!/bin/bash

# Be verbose, and stop with error as soon there's one
set -ev
# for gpaw
sudo apt-get install libxc-dev
pip install codecov
pip install -U pip setuptools wheel 

# install optional dependencies
pip install pythtb # tbmodels spglib
pip install ase
pip install gpaw==24.1.0
gpaw info
gpaw install-data --register ~/gpaw-data
pip install -U wannierberri[all]
sudo apt-get install wannier90

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
