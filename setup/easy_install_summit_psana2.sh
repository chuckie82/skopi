#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

# Setup environment.
cat > env.sh <<EOF
module load gcc/7.4.0
module load cuda/10.1.168

export PYVER=3.7

# variables needed for conda
export CONDA_PREFIX=$PWD/conda

# variables needed for psana
export LCLS2_DIR="$PWD/lcls2"

# variables needed to get the conda env
if [[ -d \$CONDA_PREFIX ]]; then
    source \$CONDA_PREFIX/etc/profile.d/conda.sh
    # This change $CONDA_PREFIX
    conda activate myenv
    export PATH=\$CONDA_PREFIX/bin:\$PATH
    export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
fi

# variables needed to run psana
if [[ -e \$LCLS2_DIR/setup_env.sh ]]; then
    export PATH="\$LCLS2_DIR/install/bin:\$PATH"
    export PYTHONPATH="\$LCLS2_DIR/install/lib/python\$PYVER/site-packages:\$PYTHONPATH"
fi
EOF

# Clean up any previous installs.
rm -rf conda
rm -rf lcls2

source env.sh

# Install Conda environment.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
bash Miniconda3-latest-Linux-ppc64le.sh -b -p $CONDA_PREFIX
rm Miniconda3-latest-Linux-ppc64le.sh
source $CONDA_PREFIX/etc/profile.d/conda.sh

PACKAGE_LIST=(
    # LCLS2 requirements:
    python=$PYVER
    cmake
    numpy
    cython
    matplotlib
    pytest=4.6
    mongodb
    pymongo
    curl
    rapidjson
    ipython
    requests
    mypy
    h5py

    # pysingfel requirements:
    numba
    scipy
    llvmlite
)

conda create -y -n myenv "${PACKAGE_LIST[@]}" -c defaults -c anaconda
conda activate myenv
conda install -y amityping -c lcls-ii
conda install -y bitstruct -c conda-forge

# Build mpi4py
CC=$OMPI_CC MPICC=mpicc pip install -v --no-binary mpi4py mpi4py

# Install Psana
git clone https://github.com/slac-lcls/lcls2.git $LCLS2_DIR
pushd $LCLS2_DIR
CC=/sw/summit/gcc/7.4.0/bin/gcc CXX=/sw/summit/gcc/7.4.0/bin/g++ ./build_all.sh -d
popd

# Install pysingfel
pushd ..
pip install -e .
popd

echo
echo "Done. Please run 'source env.sh' to use this build."
