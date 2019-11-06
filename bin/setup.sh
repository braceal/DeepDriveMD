#!/bin/bash

set -Eeuo pipefail

module load python/2.7.0
pip2 install virtualenv
virtualenv env2
source env2/bin/activate
pip2 install --upgrade pip setuptools wheel
pip2 install -r /src/entkdrive/requirements.txt
pip2 install -e /src/entkdrive/
deactivate

module load python/3.7.0
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -e .
deactivate
