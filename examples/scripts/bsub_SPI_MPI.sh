#!/bin/bash
# BEGIN LSF DIRECTIVES
#BSUB -P CHM137
#BSUB -o job-%J.out
#BSUB -e job-%J.err
#BSUB -W 02:00
#BSUB -nnodes 100
#BSUB -alloc_flags gpumps
# END LSF DIRECTIVES

t_start=`date +%s`

source $PROJWORK/chm137/adse13-198/pysingfel/setup/env.sh

export PS_PARALLEL=none
export OMP_NUM_THREADS=1
export LCLS_CALIB_HTTP=http://login2:9357/calib_ws

jsrun -n600 -a1 -g1 -c1 -dpacked --bind=packed:1 python SPI_MPI.py --pdb=../input/pdb/3iyf.pdb --geom=../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data --beam=../input/beam/amo86615.beam --numPatterns=1000000 --outDir=$PROJWORK/chm137/adse13-198/SPI_patterns

t_end=`date +%s`
echo PSJobCompleted TotalElapsed $((t_end-t_start)) $t_start $t_end
