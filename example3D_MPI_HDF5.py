import sys
import h5py as h5
import time
import pysingfel as ps
import pysingfel.gpu as pg
from mpi4py import MPI
import numpy as np
#from geometry import *
#from util import *
#from diffraction import *
#from particle import Particle
#from beam import Beam
#from ps.detector import pnccdDetector
#from util import symmpdb
#from detector import *
# TO DO: arguments and parsing
# more modularization


def main():

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sz = comm.size

    print("====================================================================")
    print("Running %d parallel MPI processes, rank=%d" % (comm.size,rank))

    # Distribute workload so that each MPI process analyzes image number i, where 
    # i % comm.size == comm.rank
    #For example if comm.size == 4:
    #     rank 0: 0, 4, 8
    #     rank 1: 1, 5, 9
    #     rank 2: 2, 6, 10
    #     rank 3: 3, 7, 11


    comm.Barrier()
    t_start = MPI.Wtime()
    particle = ps.Particle()
    if rank == 0:   
        print("Reading PDB file...")
    particle.read_pdb('/reg/neh/home5/marcgri/Software/pysingfel/examples/input/3iyf.pdb',ff='WK')
    #print particle.dtype
    #      comm.send(particle,tag=77,dest=i)
    #   elif  i==rank:
    #      particle = comm.recv(particle,tag=77,source=0);
    #      comm.Barrier()
    if rank==0:
       # reading beam and detector files
       beam= ps.Beam('/reg/neh/home5/marcgri//Software/pysingfel/examples/input/exp_chuck.beam')
       det = ps.PnccdDetector(geom='/reg/neh/home5/marcgri/Software/pysingfel/examples/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',beam=beam)
    else:
       beam = None
       det = None 
    if rank==0: 
       print("Broadcasting input to processes...")
    beam = comm.bcast(beam,root=0)
    det = comm.bcast(det,root=0)
    #comm.Barrier()
    pattern_shape = det.pedestal.shape
    pixel_momentum = det.pixel_position_reciprocal
    
    # create hdf5 file
    if rank==0:
       print("Creating HDF5 file and datasets...")    
    ff = h5.File('saveHDF5_parallel.h5','w')

    dat1 = ff.create_dataset('img',shape=(1,pattern_shape[0],pattern_shape[1],pattern_shape[2]), maxshape=(None,pattern_shape[0],pattern_shape[1],pattern_shape[2]),chunks=(1,pattern_shape[0],pattern_shape[1],pattern_shape[2]),dtype=np.int32)
    dat2 = ff.create_dataset('imgOrient',shape=(1,pattern_shape[0]),maxshape=(None,pattern_shape[0]),chunks=(1,pattern_shape[0]),dtype=np.float32)
    dat3 = ff.create_dataset('imgIndex',shape=(1,1),maxshape=(None,1),chunks=(1,1),dtype=np.int32)
    comm.Barrier()
    
    number = 100
    orientations = ps.geometry.get_uniform_quat(num_pts=number)
    
    slice_num = orientations.shape[0]
    pixel_num = np.prod(pattern_shape)

    # Create variable to hold the slices
    sliceOne = np.zeros((pattern_shape))
    mesh_length = 128
    

    mesh, voxel_length = det.get_reciprocal_mesh(voxel_number_1d=mesh_length)
    
    volume=pg.diffraction.calculate_diffraction_pattern_gpu(mesh,particle,return_type='intensity')

    rotmat = np.zeros((number,3,3))
    #rotated_pixel_position = np.zeros((number,4,512,512,3))
    rotated_pixel_position = np.zeros((1,pattern_shape[0],pattern_shape[1],pattern_shape[2],3))
    
    # set up  index and weight arrays
    index = np.zeros((1,pattern_shape[0],pattern_shape[1],pattern_shape[2],8,3),dtype='int32')
    weight= np.zeros((1,pattern_shape[0],pattern_shape[1],pattern_shape[2],8),dtype='int32')
     
    # parallel loop striping tasks across processes
    for i in range(rank,number-1 ,sz):
        
        # get rotation matrix from quaternion inputing single orientation
        rotmat[i] = ps.geometry.quaternion2rot3d(orientations[i,:])  
        
        # rotate pixels in reciprocal space
        rotated_pixel_position = ps.geometry.rotate_pixels_in_reciprocal_space(rotmat[i], pixel_momentum)
        
        # generate indices and weights
        index, weight = ps.geometry.get_weight_and_index(pixel_position=rotated_pixel_position,
                                             voxel_length=voxel_length,
                                             voxel_num_1d=volume.shape[0])
        # take individual slice
        sliceOne = ps.geometry.take_one_slice(local_index=index,
                                          local_weight=weight,
                                          volume=volume,
                                          pixel_num=pixel_num,
                                          pattern_shape=pattern_shape)

        print("Image %d on rank %d" % (i, rank))
    
        dataIndex=i
        

        # resize datasets for incremental writing 
        dat1.resize(dat1.shape[0] + 1,axis=0)   # resize image hdf5 dataset
        dat2.resize(dat2.shape[0] + 1, axis=0)  # resize orientation dataset
        dat3.resize(dat3.shape[0] + 1, axis=0)  # resize index dataset
       
        # add new data to end of datasets
        dat1[-1,:,:,:] = sliceOne        # image
        dat2[-1,:] = orientations[i,:]   # orientation
        dat3[-1,:] = dataIndex           # index
    comm.Barrier()  
    t_end = MPI.Wtime()
    

    # finish up
    if rank==0:
       print("Finishing constructing %d patterns in %f seconds" % (number,t_end - t_start ))
    #comm.Barrier()
   

if __name__ == '__main__':
    main()
   
