import sys
import h5py as h5
import time
import pysingfel as ps
import pysingfel.gpu as pg
from mpi4py import MPI
import numpy as np


def main():
    
    number = 1000
    orient = 'uniform'

    
    #params = parse_input_arguements(sys.argv)
    params = None

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sz = comm.size

    print("====================================================================")
    print("Running %d parallel MPI processes, rank=%d" % (comm.size,rank))


    if rank==0:
       t_start = MPI.Wtime()

    orientations = np.zeros((number,4))
    particle = ps.Particle()
    
    if rank == 0:
       if orient=='uniform':
          orientations = ps.geometry.get_uniform_quat(num_pts=number)
       elif orient== 'random':
          orientations = ps.geometry.get_random_quat(num_pts=number)
       else:
          orientations = None

       #particle = ps.Particle()
       print("Reading PDB file...")
       particle.read_pdb('examples/input/3iyf.pdb',ff='WK')
       # reading beam and detector files
       beam= ps.Beam('examples/input/exp_chuck.beam')
       det = ps.PnccdDetector(geom='examples/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',beam=beam)
    else:
        #beam= None
        particle = None
        det = None 
        ori= None
        print("Broadcasting input to processes...")
   
    p = comm.bcast(particle, root=0)
    #b= comm.bcast(beam,root=0)
    d = comm.bcast(det,root=0)
    ori = comm.bcast(orientations,root=0) 


    
    if rank==0:

       pattern_shape = d.pedestal.shape  
       print("Creating HDF5 file and datasets...")
       ff = h5.File('saveHDF5_parallel.h5','w')
       dat1 = ff.create_dataset('img',shape=(number,pattern_shape[0],pattern_shape[1],pattern_shape[2]),dtype=np.int32,compression="gzip", compression_opts=4)

       dat2 = ff.create_dataset('imgOrient',shape=(number,pattern_shape[0]),dtype=np.float32,compression="gzip", compression_opts=4)
    # task processing steps
    if rank == 0: # master
       for task in range(number):
           

           #  master receives signal from slave with completed task, send new task
           status1 = MPI.Status()
	   comm.recv(source=MPI.ANY_SOURCE,tag=2,status=status1)

           rnk = status1.Get_source()
           print("Rank 0 received done status from rank %d" % rnk)           
           comm.send(task,dest=rnk)
           print("Rank 0 sending task %d to rank %d" % (task,rnk))
           

           # master receives image
           status2 = MPI.Status()
           #1i = status2.Get_source()
           img = comm.recv(source=MPI.ANY_SOURCE,tag=1,status=status2)
           i = status2.Get_source()
           print("Rank 0: Received image from rank %d" % i)
           
           # put processed image and orientation into dataset
           dat1[task,:,:,:] = img
           dat2[task,:] = orientations[task,:]
           print("Task %d writing image and orientation to HDF5 dataset" % task)
       
       # when all tasks complete, send kill signal from master to each slave processss      
       for k in range(1,comm.size):
           comm.send(-1,dest=k)
           print("Rank 0 sending kill signal to rank %d" % k)

    else: # slave
        # initial calculations
        slices_num = ori.shape[0]
        pattern_shape = d.pedestal.shape
        pixel_momentum = d.pixel_position_reciprocal
        #pixel_num = np.prod(pattern_shape)
        sliceOne = np.zeros((pattern_shape))
        #rotated_pixel_position = np.zeros((1,pattern_shape[0],pattern_shape[1],pattern_shape[2],3))
        mesh_length = 128
        mesh,voxel_length = d.get_reciprocal_mesh(voxel_number_1d=mesh_length)
        volume = pg.diffraction.calculate_diffraction_pattern_gpu(mesh, p,return_type='intensity')
        
        counter = 0
        comm.send(counter,dest=0,tag=2)
        print("Sent counter, rank=%d" % rank)
        while True:
           counter = comm.recv(source=0)
           print("Received counter, rank %d" % rank)
           if counter < 0:
               print("rank %d received kill signal" % rank)
               # end of simulation
               break
           
           # transform quaternion (set of orientations) into 3D rotation  
           rotmat = ps.geometry.quaternion2rot3d(ori[counter,:])
           
           oneSlice = slave_calc_pattern(rot3d=rotmat,pixel_momentum=pixel_momentum,pattern_shape=pattern_shape,volume=volume,voxel_length=voxel_length)
           # new pixel position in reciprocal space
           #rotated_pixel_position = ps.geometry.rotate_pixels_in_reciprocal_space(rotmat, pixel_momentum)

           # generate indices and weights
           #index, weight = ps.geometry.get_weight_and_index(pixel_position=rotated_pixel_position,
           #                                  voxel_length=voxel_length,
           #                                  voxel_num_1d=volume.shape[0])
           # take one slice
           #sliceOne = ps.geometry.take_one_slice(local_index=index,
           #                              local_weight=weight,
           #                               volume=volume,
           #                               pixel_num=pixel_num,
           #                              pattern_shape=pattern_shape)

           print("Sending slice %d from rank %d" % (counter,rank))
    
           comm.send(sliceOne,dest=0,tag=1)
           print("Sending counter from rank %d" % rank)
           comm.send(counter,dest=0, tag=2)

    comm.Barrier()
    if rank==0:
       t_end = MPI.Wtime()
       print("Finishing constructing %d patterns in %f seconds" % (number,t_end - t_start ))

def slave_calc_pattern(rot3d,pixel_momentum,pattern_shape,volume,voxel_length):
    
    pixel_num = np.prod(pattern_shape)
    sliceOne = np.zeros((pattern_shape))
    rotated_pixel_position = np.zeros((1,pattern_shape[0],pattern_shape[1],pattern_shape[2],3))

    # new pixel position in reciprocal space
    rotated_pixel_position = ps.geometry.rotate_pixels_in_reciprocal_space(rot3d, pixel_momentum)

    # generate indices and weights
    index, weight = ps.geometry.get_weight_and_index(pixel_position=rotated_pixel_position,
                                             voxel_length=voxel_length,
                                             voxel_num_1d=volume.shape[0])
    oneSlice = ps.geometry.take_one_slice(local_index=index,local_weight=weight, volume=volume, pixel_num= pixel_num, pattern_shape=pattern_shape)
   
    return oneSlice

def parse_input_arguments(args):
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--PDBFile',help='PDB input file')
    parse.add_argument('--detectorFile', help='f')
    parse.add_argument('--beamFile', help='Beam file defining X-ray beam')
    parse.add_argument('--numSlices',type=int, help='Number of slices/diffraction patterns')
    parse.add_argument('--rotation', type=int, help='Uniform, random, or no orientation')
    # convert argparse to dict
    return vars(parse.parse_args(args))


if __name__ == '__main__':
    main()
   
