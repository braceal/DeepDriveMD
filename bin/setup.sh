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

# See discussion: https://stackoverflow.com/questions/42997258/virtualenv-activate-script-wont-run-in-bash-script-with-set-euo#48327176
set +u
source env2/bin/activate
set -u

pip2 install --upgrade pip virtualenv setuptools wheel
pip2 install -e /src/entkdriver/setup.py
deactivate

module load python/3.7.0
python3 -m venv env
set +u
source env/bin/activate
set -u
pip3 install --upgrade pip setuptools wheel
pip3 install -e .
deactivate
