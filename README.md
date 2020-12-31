# DeepDriveMD
Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding

---

# Important!!
This repository is out of date. Please see our latest version [here](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) or refer to our [website](https://deepdrivemd.github.io/). Thank you!

---


Implementation of: https://arxiv.org/pdf/1909.07817.pdf

Project location: https://github.com/braceal/DeepDriveMD.git

# Instructions

Initial setup on Summit https://www.olcf.ornl.gov/summit/
```
git clone https://github.com/braceal/DeepDriveMD.git
cd DeepDriveMD/

module load python/3.6.6-anaconda3-5.3.0
module load hdf5/1.10.3
module load gcc/6.4.0

. /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh

conda env create python=3.6 -p ./conda-env -f summit-env.yml
conda activate ./conda-env
```

Installing openmm from source
See http://docs.openmm.org/latest/userguide/library.html?highlight=nvcc
section 8 for additional instructions.

```
module load gcc/7.4.0  
module load cuda/10.1.168
module load cmake/3.15.2

mkdir deps
cd deps/
git clone https://github.com/pandegroup/openmm.git
git checkout b7a4960f9a497976a98993dc84f5e60d839f5f26
mkdir openmm/build_openmm
cd openmm/build_openmm/
ccmake ..
```
ccmake will open a visual configuration program

Options to set (some advanced):

```
MAKE_INSTALL_PREFIX    /<path-from-root>/DeepDriveMD/conda-env
CUDA_HOST_COMPILER     /sw/summit/gcc/7.4.0/bin/gcc
CUDA_SDK_ROOT_DIR      /sw/summit/cuda/10.1.168/samples
CUDA_TOOLKIT_ROOT_DIR  /sw/summit/cuda/10.1.168
CMAKE_CXX_COMPILER     /sw/summit/gcc/7.4.0/bin/g++
CMAKE_C_COMPILER       /sw/summit/gcc/7.4.0/bin/gcc
PYTHON_EXECUTABLE      /<path-from-root>/DeepDriveMD/conda-env/bin/python
SWIG_EXECUTABLE        /<path-from-root>/DeepDriveMD/conda-env/bin/swig
```

Continue to configure until there are no remaining errors, then generate.
Note: make sure to double check the settings, especially MAKE_INSTALL_PREFIX, after
      configuring for the first time and once again after generation.

Steps to compile (Run from build_openmm directory):
```
make -j 42
make install
make PythonInstall
```

The full test suite can be run with:
```
make test
```

The python API can be tested with:
```
python -m simtk.testInstallation
```

The python test should output:
```
OpenMM Version: 7.5
Git Revision: b7a4960f9a497976a98993dc84f5e60d839f5f26

There are 3 Platforms available:

1 Reference - Successfully computed forces
2 CPU - Successfully computed forces
3 CUDA - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU: 1.93944e-06
Reference vs. CUDA: 6.73051e-06
CPU vs. CUDA: 6.21163e-06

All differences are within tolerance.
```


