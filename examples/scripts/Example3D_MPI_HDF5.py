import os, sys
import h5py as h5
import time
import pysingfel as ps
import pysingfel.gpu as pg
from mpi4py import MPI
import numpy as np
import argparse

# Example: mpirun -n 2 python Example3D_MPI_HDF5.py --pdb=../input/3iyf.pdb --geom=../lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data --beam=../input/exp_chuck.beam --numSlices=1 --UniformOrientation=1

def main():
    # Parse user input
    params = parse_input_arguments(sys.argv)
    pdb = params['pdb']
    geom = params['geom']
    beam = params['beam']
    orient = int(params['UniformOrientation'])
    number = int(params['numSlices'])
    outDir = params['outDir']
 
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sz = comm.size

    det = None
    data = None
    if rank==0:
       print("====================================================================")
       print("Running %d parallel MPI processes" % (comm.size))

       t_start = MPI.Wtime()
         
       orientations = np.zeros((number,4))
       particle = ps.Particle()

    if rank==0:
       if orient== 1:
          orientations = ps.geometry.get_uniform_quat(num_pts=number).astype(np.float32)
       elif orient== 0:
          orientations = ps.geometry.get_random_quat(num_pts=number).astype(np.float32)

       print("Reading PDB file...")
       particle.read_pdb(pdb, ff='WK')
       # reading beam and detector files
       beam= ps.Beam(beam)
       det = ps.PnccdDetector(geom=geom, beam=beam)
       print("Broadcasting input to processes...")
    
       data = {'particle': particle, 'orientations': orientations, 'detector': det}

    dct = comm.bcast(data,root=0)
    
    if rank==0:
       pattern_shape = det.pedestal.shape  
       f = h5.File(os.path.join(outDir,'saveHDF5_parallel.h5'),'w')
       dset = f.create_dataset('img', shape=(number,)+pattern_shape,dtype=np.int32, chunks=(1,)+pattern_shape, compression="gzip", compression_opts=4)
       f.create_dataset('orientation', data=orientations, compression="gzip", compression_opts=4)
       print("Done creating HDF5 file and datasets...")

       c = 0
       while c < number:
           status1 = MPI.Status()
           result = comm.recv(source=MPI.ANY_SOURCE,status=status1) # (index,photImg) 
           i = status1.Get_source()
           print("Rank 0: Received image %d from rank %d" % (result[0],i)) 
           dset[result[0],:,:,:] = result[1]
           c += 1

    else: # slave
        # initialize intensity volume
        ori = dct['orientations']
        det = dct['detector']
        particle = dct['particle']
        slices_num = ori.shape[0]
        pattern_shape = det.pedestal.shape
        pixel_momentum = det.pixel_position_reciprocal
        sliceOne = np.zeros((pattern_shape))
        mesh_length = 128
        mesh,voxel_length = det.get_reciprocal_mesh(voxel_number_1d=mesh_length)
        intensVol = pg.diffraction.calculate_diffraction_pattern_gpu(mesh, particle, return_type='intensity')

        for i in range((rank-1),number,sz-1):
           # transform quaternion (set of orientations) into 3D rotation  
           rotmat = ps.geometry.quaternion2rot3d(ori[i,:])
           
           intensSlice = slave_calc_intensity(rot3d = rotmat,
                                         pixel_momentum = pixel_momentum,
                                         pattern_shape = pattern_shape,
                                         volume = intensVol,
                                         voxel_length = voxel_length)

           # Convert the one image to photons 
           photImg = det.add_correction_and_quantization(pattern=intensSlice).astype(np.int32)

           print("Sending slice %d from rank %d" % (i,rank))
           comm.send((i,photImg),dest=0)

    if rank==0:
       t_end = MPI.Wtime()
       print("Finishing constructing %d patterns in %f seconds" % (number,t_end-t_start))

       import matplotlib.pyplot as plt
       # Display first diffraction image
       photImgAssem = det.assemble_image_stack(image_stack=f['img'][0,:,:,:])
       plt.imshow(photImgAssem, interpolation='none', vmin=0,vmax=4)
       plt.colorbar()
       plt.show()
       f.close()


def slave_calc_intensity(rot3d, pixel_momentum, pattern_shape, volume, voxel_length):
    """
    Take an Ewald slice at a given orientation from a diffracted intensity volume

    :param rot3d:
    :param pixel_momentum:
    :param pattern_shape:
    :param volume:
    :param voxel_length:
    :return: 
    """
    pixel_num = np.prod(pattern_shape)
    sliceOne = np.zeros((pattern_shape))
    rotated_pixel_position = np.zeros((1,)+pattern_shape+(3,))

    # new pixel position in reciprocal space
    rotated_pixel_position = ps.geometry.rotate_pixels_in_reciprocal_space(rot3d, pixel_momentum)

    # generate indices and weights
    index, weight = ps.geometry.get_weight_and_index(pixel_position=rotated_pixel_position,
                                             voxel_length=voxel_length,
                                             voxel_num_1d=volume.shape[0])
    intensSlice = ps.geometry.take_one_slice(local_index = index,
                                          local_weight = weight, 
                                          volume = volume, 
                                          pixel_num = pixel_num, 
                                          pattern_shape = pattern_shape)
    return intensSlice

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--pdb', type=str, help='PDB file')
    parse.add_argument('-g', '--geom', type=str, help='Psana geometry file')
    parse.add_argument('-b', '--beam', type=str, help='Beam file defining X-ray beam')
    parse.add_argument('-n', '--numSlices',type=int, help='Number of slices/diffraction patterns')
    parse.add_argument('-u', '--UniformOrientation', type=int, help='Uniform (1), random (0)')
    parse.add_argument('-o', '--outDir', default='', type=str, help='output directory')
    # convert argparse to dict
    return vars(parse.parse_args(args))


if __name__ == '__main__':
    main()
   

