import os
import sys
import time
import numpy as np
import scipy as sp
import h5py as h5
from mpi4py import MPI # module required to use MPI
import argparse
import skopi as sk
from skopi.util import asnumpy, xp


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
    numParticles = int(params['numParticles'])
    numPatterns = int(params['numPatterns'])
    outDir = params['outDir']
    saveName = params['saveNameHDF5']

    data = None

    if rank == 0:
        print ("====================================================================")
        print ("Running %d parallel MPI processes" % size)

        t_start = MPI.Wtime()

        # load beam
        beam = sk.Beam(beam)

        # load and initialize the detector
        det = sk.PnccdDetector(geom=geom, beam=beam)

        # create particle object(s)
        particle = sk.Particle()
        particle.read_pdb(pdb, ff='WK')

        experiment = sk.SPIExperiment(det, beam, particle)

        f = h5.File(os.path.join(outDir,"SPI_MPI.h5"),"w")
        f.attrs['numParticles'] = numParticles
        experiment.volumes[0] = xp.asarray(experiment.volumes[0])
        experiment.volumes[0] = xp.asarray(experiment.volumes[0])
        dset_volume = f.create_dataset("volume", data=experiment.volumes[0], compression="gzip", compression_opts=4)

        data = {"detector": det, "beam": beam, "particle": particle}
        print ("Broadcasting input to processes...")

    dct = comm.bcast(data,root=0)

    if rank == 0:
        pattern_shape = det.pedestals.shape # (4, 512, 512)

        dset_intensities = f.create_dataset("intensities", shape=(numPatterns,)+pattern_shape, dtype=np.float32, chunks=(1,)+pattern_shape, compression="gzip", compression_opts=4) # (numPatterns, 4, 512, 512)
        dset_photons = f.create_dataset("photons", shape=(numPatterns,)+pattern_shape, dtype=np.float32, chunks=(1,)+pattern_shape, compression="gzip", compression_opts=4)
        dset_positions = f.create_dataset("positions", shape=(numPatterns,)+(numParticles, 3), dtype=np.float32, chunks=(1,)+(numParticles, 3), compression="gzip", compression_opts=4)
        dset_orientations = f.create_dataset("orientations", shape=(numPatterns,)+(numParticles, 4), chunks=(1,)+(numParticles, 4), compression="gzip", compression_opts=4)
        dset_pixel_index_map = f.create_dataset("pixel_index_map", data=det.pixel_index_map, compression="gzip", compression_opts=4)
        dset_pixel_position_reciprocal = f.create_dataset("pixel_position_reciprocal", data=det.pixel_position_reciprocal, compression="gzip", compression_opts=4)

        print ("Done creating HDF5 file and dataset...")

        n = 0
        while n < numPatterns:
            status1 = MPI.Status()
            (ind, img_slice_intensities, img_slice_positions, img_slice_orientations) = comm.recv(source=MPI.ANY_SOURCE,status=status1)
            i = status1.Get_source()
            print ("Rank 0: Received image %d from rank %d" % (ind,i))
            dset_intensities[ind,:,:,:] = np.asarray(img_slice_intensities)
            dset_photons[ind,:,:,:] = det.add_quantization(img_slice_intensities)
            dset_positions[ind,:,:] = np.asarray(img_slice_positions)
            dset_orientations[ind,:,:] = np.asarray(img_slice_orientations)
            n += 1
    else:
        det = dct['detector']
        beam = dct['beam']
        particle = dct['particle']

        experiment = sk.SPIExperiment(det, beam, particle)
        for i in range((rank-1),numPatterns,size-1):
            img_slice = experiment.generate_image_stack(return_intensities=True, return_positions=True, return_orientations=True, always_tuple=True)
            img_slice_intensities = img_slice[0]
            img_slice_positions = img_slice[1]
            img_slice_orientations = img_slice[2]
            comm.ssend((i,img_slice_intensities,img_slice_positions,img_slice_orientations),dest=0)

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
    parse.add_argument('-m', '--numParticles', type=int, help='number of particles')
    parse.add_argument('-n', '--numPatterns', type=int, help='number of diffraction patterns')
    parse.add_argument('-o', '--outDir', default='', type=str, help='output directory')
    parse.add_argument('-s', '--saveNameHDF5',default='saveHDF5_parallel.h5',type=str, help='filename for image dataset')
    # convert argparse to dict
    return vars(parse.parse_args(args))

if __name__ == '__main__':
    main()
