# DeepDriveMD
Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding

Implementation of: https://arxiv.org/pdf/1909.07817.pdf

Project location: https://github.com/braceal/DeepDriveMD.git

# Instructions

Initial setup on Summit https://www.olcf.ornl.gov/summit/
```
git clone https://github.com/braceal/DeepDriveMD.git
cd DeepDriveMD/

module load python/2.7.15
module load py-setuptools/40.4.3-py2
module load py-virtualenv/16.0.0-py2

virtualenv env2
source env2/bin/activate
pip2 install --upgrade pip setuptools wheel
pip2 install -e src/entkdriver/
deactivate 

module load python/3.7.0
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -e .

```
Now the python3 virtual environment is enabled.

To deactivate python3 virtual environment:
```
deactivate
```
