# DeepDriveMD
Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding

Implementation of: https://arxiv.org/pdf/1909.07817.pdf

Project location: https://github.com/braceal/DeepDriveMD.git

# Instructions

Initial setup on Summit https://www.olcf.ornl.gov/summit/
```
git clone https://github.com/braceal/DeepDriveMD.git
cd DeepDriveMD/

module load python/3.7.0
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

This project uses openmm for molecular dynamics simulations which requires anaconda to install.
First update your conda environment:
```
conda update conda
```
Then install openmm:
```
conda install -c omnia openmm-setup
```
