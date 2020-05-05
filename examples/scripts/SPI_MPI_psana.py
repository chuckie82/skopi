import sys
import os
import time
import numpy as np
import scipy as sp
import h5py as h5
from mpi4py import MPI # module required to use MPI
import argparse

import pysingfel as ps
import pysingfel.gpu as pg
from pysingfel.util import asnumpy, xp

ROOT_DIR=os.environ["PYSINGFEL_DIR"]
sys.path.append("/ccs/home/iris/adse13-198/pysingfel/setup/lcls2/psana")

# Example: mpirun -n 2 python SPI_MPI_psana.py --pdb=../input/pdb/3iyf.pdb --geom=../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data --beam=../input/beam/amo86615.beam --numPatterns=60 --outDir=../output

# set up MPI environment
comm = MPI.COMM_WORLD # communication module
size = comm.Get_size() # number of processors available
rank = comm.Get_rank() # the rank of the process (the rank of a process always ranges from 0 to size-1)

def main():
    # parse user input
    params = parse_input_arguments(sys.argv)
    pdb = params['pdb']
    geom = params['geom']
    beam = params['beam']
    numPatterns = int(params['numPatterns'])
    outDir = params['outDir']
    saveName = params['saveNameHDF5']

    data = None

    if rank == 0:
        print ("====================================================================")
        print ("Running %d parallel MPI processes" % size)

        t_start = MPI.Wtime()

        # load beam
        beam = ps.Beam(beam)

        # load and initialize the detector
        det = ps.PnccdDetector(geom=geom, beam=beam)

        # create particle object(s)
        particle = ps.Particle()
        particle.read_pdb(pdb, ff='WK')

        data = {"detector": det, "beam": beam, "particle": particle}
        print ("Broadcasting input to processes...")

    dct = comm.bcast(data,root=0)

    if rank == 0:
        pattern_shape = det.pedestals.shape # (4, 512, 512)

        f = h5.File(os.path.join(outDir,"SPI_MPI.h5"),"w")
        dset = f.create_dataset("intensity", shape=(numPatterns,)+pattern_shape, dtype=np.float32, chunks=(1,)+pattern_shape, compression="gzip", compression_opts=4) # (numPatterns, 4, 512, 512)

        print ("Done creating HDF5 file and dataset...")

        n = 0
        while n < numPatterns:
            status1 = MPI.Status()
            (ind,img) = comm.recv(source=MPI.ANY_SOURCE,status=status1) # (index,photImg) 
            i = status1.Get_source()
            print ("Rank 0: Received image %d from rank %d" % (ind,i)) 
            dset[ind,:,:,:] = img
            n += 1
    else:
        det = dct['detector']
        beam = dct['beam']
        particle = dct['particle']
        experiment = ps.SPIExperiment(det, beam, particle)
        for i in range((rank-1),numPatterns,size-1):
            img_intensity = experiment.generate_image_stack()
            print ("Sending slice %d from rank %d" % (i,rank))
            comm.ssend((i,img_intensity),dest=0)

    if rank == 0:
        t_end = MPI.Wtime()
        print ("Finishing constructing %d patterns in %f seconds" % (numPatterns,t_end-t_start))
        f.close()
        sys.exit()

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--pdb', type=str, help='PDB file')
    parse.add_argument('-g', '--geom', type=str, help='psana geometry file')
    parse.add_argument('-b', '--beam', type=str, help='beam file defining X-ray beam')
    parse.add_argument('-n', '--numPatterns',type=int, help='number of diffraction patterns')
    parse.add_argument('-o', '--outDir', default='', type=str, help='output directory')
    parse.add_argument('-s', '--saveNameHDF5',default='saveHDF5_parallel.h5',type=str, help='filename for image dataset')
    # convert argparse to dict
    return vars(parse.parse_args(args))

if __name__ == '__main__':
    main()
