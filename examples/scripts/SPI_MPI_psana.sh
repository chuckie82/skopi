#!/bin/bash
#
#SBATCH --job-name=SPI_MPI
#SBATCH --output=spi_mpi.out
#SBATCH --partition=anagpu
#SBATCH --gres=gpu:1080ti:1
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=4
#SBATCH --mail-user=user@slac.stanford.edu
#SBATCH --mail-type=ALL

PYSINGFEL_DIR=os.environ["PYSINGFEL_DIR"]

# source the psana2 environment

srun python SPI_MPI.py --pdb=$PYSINGFEL_DIR/input/pdb/3iyf.pdb --geom=$PYSINGFEL_DIR/input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data --beam=$PYSINGFEL_DIR/input/beam/amo86615.beam --numPatterns=1000 --outDir=$PYSINGFEL_DIR/output
