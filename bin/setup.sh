#!/bin/bash

set -Eeuo pipefail

echo "module load python/2.7.15"
module load python/2.7.15
echo "module load py-setuptools/40.4.3-py2"
module load py-setuptools/40.4.3-py2
echo "module load py-virtualenv/16.0.0-py2"
module load py-virtualenv/16.0.0-py2

echo "virtualenv env2"
virtualenv env2

source env2/bin/activate
pip2 install --upgrade pip setuptools wheel
pip2 install -e src/entkdrive/
deactivate

module load python/3.7.0
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -e .
deactivate
