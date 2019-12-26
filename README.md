# DeepDriveMD
Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding

Implementation of: https://arxiv.org/pdf/1909.07817.pdf

Project location: https://github.com/braceal/DeepDriveMD.git

# Instructions
This project uses openmm for molecular dynamics simulations which requires anaconda to install.
First update your conda environment:
```
conda update conda
```

Initial setup on Summit https://www.olcf.ornl.gov/summit/
```
git clone https://github.com/braceal/DeepDriveMD.git
cd DeepDriveMD/

module load python/3.7.0-anaconda3-5.3.0
conda env create python=3.7 -p ./conda-env -f environment.yml
conda activate ./conda-env
#conda install -p ./conda-env -c omnia openmm-setup
```

Installing openmm from source
See http://docs.openmm.org/latest/userguide/library.html?highlight=nvcc
section 8 for additional instructions.

```
module load gcc/7.4.0  
module load cuda/9.2.148 
module load cmake/3.15.2

mkdir -p ./deps/build_openmm
cd deps/
git clone https://github.com/pandegroup/openmm.git
cd build_openmm
ccmake -i ../openmm
```
ccmake will open a visual configuration program

Options to set:

    CUDA_HOST_COMPILER=gcc 7.4
    CMAKE_INSTALL_PREFIX=<path-from-root>/DeepDriveMD/conda-env

Continue to configure until there are no remaining errors, then generate.
