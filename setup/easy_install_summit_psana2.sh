#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

# Setup environment.
cat > env.sh <<EOF
module load gcc/7.5.0
module load cuda/10.2.89
export CC=gcc
export CXX=g++

# compilers for mpi4py
export MPI4PY_CC=\$OMPI_CC
export MPI4PY_MPICC=mpicc

export PYVER=3.8
export CUDA_HOME=\$OLCF_CUDA_ROOT
export CUPY_DIR="$PWD/cupy"

# variables needed for conda
export CONDA_PREFIX=$PWD/conda

# variables needed for psana
export LCLS2_DIR="$PWD/lcls2"
export PATH="\$LCLS2_DIR/install/bin:\$PATH"
export PYTHONPATH="\$LCLS2_DIR/install/lib/python\$PYVER/site-packages:\$PYTHONPATH"
export PS_PARALLEL=none

# variables needed to get the conda env
if [[ -d \$CONDA_PREFIX ]]; then
    source \$CONDA_PREFIX/etc/profile.d/conda.sh
    # This change $CONDA_PREFIX
    conda activate myenv
    export PATH=\$CONDA_PREFIX/bin:\$PATH
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
fi
EOF

# Clean up any previous installs.
rm -rf conda
rm -rf lcls2
rm -rf cupy

source env.sh

# Install Conda environment.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
bash Miniconda3-latest-Linux-ppc64le.sh -b -p $CONDA_PREFIX
rm Miniconda3-latest-Linux-ppc64le.sh
source $CONDA_PREFIX/etc/profile.d/conda.sh

PACKAGE_LIST=(
    python=$PYVER
    matplotlib
    numpy
    scipy
    pytest
    h5py

    # lcls2
    setuptools=46.4.0  # temp need specific version
    cmake
    cython
    mongodb
    pymongo
    curl
    rapidjson
    ipython
    requests
    mypy
    prometheus_client

    # cupy requirements:
    fastrlock

    # skopi requirements:
    numba
    scikit-learn

    # convenience
    tqdm  
)

conda create -y -n myenv "${PACKAGE_LIST[@]}" -c defaults -c anaconda
conda activate myenv
conda install -y amityping -c lcls-ii
conda install -y bitstruct krtc -c conda-forge

# Build mpi4py
CC=$OMPI_CC MPICC=mpicc pip install -v --no-binary mpi4py mpi4py


CC=$MPI4PY_CC MPICC=$MPI4PY_MPICC pip install -v --no-binary mpi4py mpi4py


# Install cupy
git clone https://github.com/cupy/cupy.git $CUPY_DIR
pushd $CUPY_DIR
git submodule update --init
pip install --no-cache-dir .
popd

# Install Psana
git clone https://github.com/slac-lcls/lcls2.git $LCLS2_DIR
pushd $LCLS2_DIR
./build_all.sh -d
popd

# Install skopi
pushd ..
pip install -e .
popd

echo
echo "Done. Please run 'source env.sh' to use this build."
