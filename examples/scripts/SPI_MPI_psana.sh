#!/bin/bash
#
#SBATCH --job-name=SPI_MPI
#SBATCH --output=spi_mpi.out
#SBATCH --partition=anagpu
#SBATCH --gres=gpu:1080ti:1
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=4
#SBATCH --mail-user=iris@slac.stanford.edu
#SBATCH --mail-type=ALL


CWD=/reg/neh/home/iris/Software/pysingfel/examples
source ~iris/lcls2/setup_env.sh
export PYTHONPATH=/reg/neh/home/iris/Software/pysingfel:/reg/neh/home/iris/lcls2/install/lib/python3.7/site-packages:$PYTHONPATH
srun python SPI_MPI.py --pdb=$CWD/input/pdb/3iyf.pdb --geom=$CWD/input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data --beam=$CWD/input/beam/amo86615.beam --numPatterns=1000 --outDir=$CWD/output
