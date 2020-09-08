"""
Deeban Ramalingam (deebanr@slac.stanford.edu)

This script implements a hybrid CPU-GPU approach to efficiently generate a synthetic dataset of diffraction patterns and their associated orientations as an HDF5 file using both the CPU and GPU. This approach uses the Message Passing Interface (MPI) communication model. In this communication model, there are three types of processes: the Master rank, the CPU ranks, and the GPU rank. Using the PDB, beamline instrument, and detector geometry, the GPU rank calculates and broadcasts the diffraction volume of the particle to the CPU ranks. The Master rank generates orientations as uniformly distributed quaternions and saves these orientations to HDF5 file. The CPU and GPU ranks repeatedly query the Master rank for batches of the orientations. The CPU and GPU ranks then use the orientations and diffraction volume to produce diffraction images, which are also added to the HDF5 file.

How to run this script:

mpiexec -n <number of processors> python cspi_generate_synthetic_dataset_double-hit_mpi_hybrid.py --config <path to Config file> --dataset <alias for the dataset>

Example on how to run this script:

mpiexec -n 16 python cspi_generate_synthetic_dataset_double-hit_mpi_hybrid.py --config cspi_generate_synthetic_dataset_config.json --dataset 3iyf-10K

Tips on using this script:

1. For an example config file, look at: pysingfel/examples/scripts/cspi_generate_synthetic_dataset_config.json

2. Use the previously defined datasets in this file to add or modify an existing dataset of your choice.

If you wish to use the Latent Space Visualizer to visualize the synthetic dataset, make the image output directory accessible to JupyterHub on PSWWW after running the script.

Example on how to make the image output directory accessible to the Latent Space Visualizer:

ln -s /reg/data/ana03/scratch/deebanr/3iyf-10K /reg/neh/home/deebanr/3iyf-10K

Tips on creating file system links with the ln command from the Terminal:

https://www.linode.com/docs/tools-reference/tools/create-file-system-links-with-ln/#use-cases-for-symbolic-links

MPI Communication Model adapted from: https://github.com/AntoineDujardin/pysingfel/blob/mpi-gpu/examples/scripts/SPI_2CEX_MPI_hybrid.py

The parameter n specifies the number of ranks. This script requires at least 2 ranks. Rank 0 is the Master Rank and Rank 1 is the GPU Rank. All other ranks are CPU ranks.
"""

# MPI parameters
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.size

if RANK == 0:
    assert N_RANKS >= 2, "This script is planned for at least 2 ranks."

MASTER_RANK = 0
GPU_RANKS = (1,)

import time
import os

# Only rank 1 uses the GPU/cupy.
os.environ["USE_CUPY"] = '1' if RANK in GPU_RANKS else '0'

# Unlock parallel but non-MPI HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

import sys

ROOT_DIR = "/reg/neh/home5/deebanr/rdeeban-pysingfel/pysingfel"
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.append(ROOT_DIR+"/../../lcls2/psana")

import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import h5py as h5

import pysingfel as ps
from pysingfel.util import asnumpy, xp


def main():
    # Parse user input for config file and dataset name
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input['config']
    dataset_name = user_input['dataset']   
    
    # Get the Config file parameters
    with open(config_file) as config_file:
        config_params = json.load(config_file)

    # Check if dataset in Config file
    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    # Get the dataset parameters from Config file parameters
    dataset_params = config_params[dataset_name]

    # Get the input dataset parameters
    pdb_file = dataset_params["pdb"]
    beam_file = dataset_params["beam"]
    beam_fluence_increase_factor = dataset_params["beamFluenceIncreaseFactor"]
    geom_file = dataset_params["geom"]
    dataset_size = dataset_params["numPatterns"]

    # Divide up the task of creating the dataset to be executed simultaneously by multiple ranks
    batch_size = dataset_params["batchSize"]

    # Get the output dataset parameters
    img_dir = dataset_params["imgDir"]
    output_dir = dataset_params["outDir"]

    # raise exception if batch_size does not divide into dataset_size
    if dataset_size % batch_size != 0:
        if RANK == MASTER_RANK:
            raise ValueError("(Master) batch_size {} should divide dataset_size {}.".format(batch_size, dataset_size))
        else:
            sys.exit(1)

    # Compute number of batches to process
    n_batches = dataset_size // batch_size

    # Flags
    save_volume = False
    with_intensities = False
    given_orientations = True
    given_positions = True

    # Constants
    photons_dtype = np.uint8
    photons_max = np.iinfo(photons_dtype).max

    # Load beam parameters
    beam = ps.Beam(beam_file)

    # Increase the beam fluence
    if not np.isclose(beam_fluence_increase_factor, 1.0):
        beam.set_photons_per_pulse(beam_fluence_increase_factor * beam.get_photons_per_pulse())

    # Load geometry of detector
    det = ps.PnccdDetector(geom=geom_file, beam=beam)

    # Get the shape of the diffraction pattern
    diffraction_pattern_height = det.detector_pixel_num_x.item()
    diffraction_pattern_width = det.detector_pixel_num_y.item()

    # Define path to output HDF5 file
    output_file = get_output_file_name(dataset_name, dataset_size, diffraction_pattern_height, diffraction_pattern_width)    
    cspi_synthetic_dataset_file = os.path.join(output_dir, output_file)

    # Generate orientations for both particles
    if given_orientations and RANK == MASTER_RANK:
        print("(Master) Generate {} orientations".format(dataset_size))
        
        # Generate orientations for the first particle
        first_particle_orientations = ps.get_uniform_quat(dataset_size, True)

        # Generate orientations for the second particle
        second_particle_orientations = ps.get_random_quat(dataset_size)

        # Assemble the orientations for both particles
        first_particle_orientations_ = first_particle_orientations[np.newaxis]
        second_particle_orientations_ = second_particle_orientations[np.newaxis]
        orientations_ = np.concatenate((first_particle_orientations_, second_particle_orientations_))
        orientations = np.transpose(orientations_, (1, 0, 2))
        
    # Generate positions for both particles
    if given_positions and RANK == MASTER_RANK:
        print("(Master) Generate {} positions".format(dataset_size))
        
        # Generate positions for the first particle
        first_particle_positions = np.zeros((dataset_size, 3))

        # Generate positions for the second particle
        second_particle_positions = generate_positions_for_second_particle(dataset_size, 2e-8, 5e-8)

        # Assemble the positions for both particles
        first_particle_positions_ = first_particle_positions[np.newaxis]
        second_particle_positions_ = second_particle_positions[np.newaxis]
        positions_ = np.concatenate((first_particle_positions_, second_particle_positions_))
        positions = np.transpose(positions_, (1, 0, 2))

    sys.stdout.flush()

    # Create a particle object
    if RANK == GPU_RANKS[0]:
        
        # Load PDB
        print("(GPU 0) Reading PDB file: {}".format(pdb_file))
        particle = ps.Particle()
        particle.read_pdb(pdb_file, ff='WK')

        # Calculate diffraction volume
        print("(GPU 0) Calculating diffraction volume")
        experiment = ps.SPIExperiment(det, beam, particle)

    else:
        experiment = ps.SPIExperiment(det, beam, None)

    sys.stdout.flush()

    # Transfer diffraction volume to CPU memory
    buffer = asnumpy(experiment.volumes[0])

    # GPU rank broadcasts diffraction volume to other ranks
    COMM.Bcast(buffer, root=1)

    # This condition is necessary if the script is run on more than one machine (each machine having 1 GPU and 9 CPU)
    if RANK in GPU_RANKS[1:]:
        experiment.volumes[0] = xp.asarray(experiment.volumes[0])

    if RANK == MASTER_RANK:
        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            print("(Master) Creating output directory: {}".format(output_dir))
            os.makedirs(output_dir)

        # Create image directory if it does not exist
        if not os.path.exists(img_dir):
            print("(Master) Creating image output directory: {}".format(img_dir))
            os.makedirs(img_dir)

        print("(Master) Creating HDF5 file to store the datasets: {}".format(cspi_synthetic_dataset_file))
        f = h5.File(cspi_synthetic_dataset_file, "w")
        
        f.create_dataset("pixel_position_reciprocal", data=det.pixel_position_reciprocal)
        f.create_dataset("pixel_index_map", data=det.pixel_index_map)
        
        if given_orientations:
            f.create_dataset("orientations", data=orientations)
        
        if given_positions:
            f.create_dataset("positions", data=positions)
        
        f.create_dataset("photons", (dataset_size, 4, 512, 512), photons_dtype)

        # Create a dataset to store the diffraction patterns
        f.create_dataset("diffraction_patterns", (dataset_size, diffraction_pattern_height, diffraction_pattern_width), dtype='f')

        if save_volume:
            f.create_dataset("volume", data=experiment.volumes[0])
        
        if with_intensities:
            f.create_dataset("intensities", (dataset_size, 4, 512, 512), np.float32)
        
        f.close()

    sys.stdout.flush()

    # Make sure file is created before others open it
    COMM.barrier()

    # Add the atomic coordinates of the particle to the HDF5 file
    if RANK == GPU_RANKS[0]:
        atomic_coordinates = particle.atom_pos

        f = h5.File(cspi_synthetic_dataset_file, "a")

        dset_atomic_coordinates = f.create_dataset("atomic_coordinates", atomic_coordinates.shape, dtype='f')
        dset_atomic_coordinates[...] = atomic_coordinates

        f.close()

    # Make sure file is closed before others open it
    COMM.barrier()
        
    # Keep track of the number of images processed
    n_images_processed = 0

    if RANK == MASTER_RANK:

        # Send batch numbers to non-Master ranks
        for batch_n in tqdm(range(n_batches)):
            
            # Receive query for batch number from a rank
            i_rank = COMM.recv(source=MPI.ANY_SOURCE)
            
            # Send batch number to that rank
            COMM.send(batch_n, dest=i_rank)
            
            # Send orientations
            if given_orientations:
                batch_start = batch_n * batch_size
                batch_end = (batch_n+1) * batch_size
                COMM.send(orientations[batch_start:batch_end], dest=i_rank)
                
            # Send positions as well
            if given_positions:
                batch_start = batch_n * batch_size
                batch_end = (batch_n+1) * batch_size
                COMM.send(positions[batch_start:batch_end], dest=i_rank)

        # Tell non-Master ranks to stop asking for more data since there are no more batches to process
        for _ in range(N_RANKS - 1):
            # Send one "None" to each rank as final flag
            i_rank = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(None, dest=i_rank)

    else:
        # Get the HDF5 file
        f = h5.File(cspi_synthetic_dataset_file, "r+")

        # Get the dataset used to store the photons
        h5_photons = f["photons"]

        # Get the dataset used to store the diffraction patterns
        h5_diffraction_patterns = f["diffraction_patterns"]

        # Get the dataset used to store intensities
        if with_intensities:
            h5_intensities = f["intensities"]

        while True:
            # Ask for batch number from Master rank
            COMM.send(RANK, dest=MASTER_RANK)

            # Receive batch number from Master rank
            batch_n = COMM.recv(source=MASTER_RANK)

            # If batch number is final flag, stop
            if batch_n is None:
                break

            # Receive orientations from Master rank
            if given_orientations:
                orientations = COMM.recv(source=MASTER_RANK)
                experiment.set_orientations(orientations)
                
            # Receive positions as well from Master rank
            if given_positions:
                positions = COMM.recv(source=MASTER_RANK)
                experiment.set_positions(positions)

            # Define a Numpy array to hold a batch of photons
            np_photons = np.zeros((batch_size, 4, 512, 512), photons_dtype)

            # Define a Numpy array to hold a batch of diffraction patterns
            np_diffraction_patterns = np.zeros((batch_size, diffraction_pattern_height, diffraction_pattern_width))

            # Define a Numpy array to hold a batch of intensities
            if with_intensities:
                np_intensities = np.zeros((batch_size, 4, 512, 512), np.float32)

            # Define the batch start and end offsets
            batch_start = batch_n * batch_size
            batch_end = (batch_n + 1) * batch_size

            # Generate batch of snapshots
            for i in range(batch_size):
                
                # Generate the image stack for the double-particle hit scenario
                image_stack_tuple = experiment.generate_image_stack(return_photons=True, return_intensities=with_intensities, always_tuple=True, multi_particle_hit=True)
                
                # Photons
                photons = image_stack_tuple[0]

                # # Raise exception if photon max exceeds max of uint8
                # if photons.max() > photons_max:
                #     raise RuntimeError("Value of photons too large for type {}.".format(photons_dtype))

                np_photons[i] = asnumpy(photons.astype(photons_dtype))

                # Assemble the image stack into a 2D diffraction pattern
                np_diffraction_pattern = experiment.det.assemble_image_stack(image_stack_tuple)

                # Add the assembled diffraction pattern to the batch
                np_diffraction_patterns[i] = np_diffraction_pattern

                # Save diffraction pattern as PNG file
                data_index = batch_start + i
                save_diffraction_pattern_as_image(data_index, img_dir, np_diffraction_pattern)

                # Intensities
                if with_intensities:
                    np_intensities[i] = asnumpy(image_stack_tuple[1].astype(np.float32))

                # Update the number of images processed
                n_images_processed += 1

            # Add the batch of photons to the HDF5 file
            h5_photons[batch_start:batch_end] = np_photons

            # Add the batch of diffraction patterns to the HDF5 file
            h5_diffraction_patterns[batch_start:batch_end] = np_diffraction_patterns

            if with_intensities:
                h5_intensities[batch_start:batch_end] = np_intensities

        # Close the HDF5 file
        f.close()

    sys.stdout.flush()

    # Wait for ranks to finish
    COMM.barrier()

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: two-atoms, 3iyf')

    # convert argparse to dict
    return vars(parse.parse_args(args))

def generate_positions_for_second_particle(n_positions_for_second_particle, lower_boundary, upper_boundary):
    positions_for_second_particle = np.zeros((n_positions_for_second_particle, 3))
    position_idx = 0
    while position_idx < n_positions_for_second_particle:
        # Adapted from: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        x_position = np.random.uniform(low=-upper_boundary, high=upper_boundary)
        y_position = np.random.uniform(low=-upper_boundary, high=upper_boundary)
        z_position = np.random.uniform(low=-upper_boundary, high=upper_boundary)
        position_for_second_particle = np.array([x_position, y_position, z_position])
        distance_from_origin = np.linalg.norm(position_for_second_particle)
        if lower_boundary < distance_from_origin and distance_from_origin < upper_boundary:
            positions_for_second_particle[position_idx] = position_for_second_particle
            position_idx += 1
    return positions_for_second_particle

def save_diffraction_pattern_as_image(data_index, img_dir, diffraction_pattern):
    img_file = 'diffraction-pattern-{}.png'.format(data_index)
    img_path = os.path.join(img_dir, img_file)
    
    im = gnp2im(diffraction_pattern)
    im.save(img_path, format='png')

def gnp2im(image_np):
    """
    Converts an image stored as a 2-D grayscale Numpy array into a PIL image.
    """
    rescaled = (255.0 / image_np.max() * (image_np - image_np.min())).astype(np.uint8)
    im = Image.fromarray(rescaled, mode='L')
    return im

def get_output_file_name(dataset_name, dataset_size, diffraction_pattern_height, diffraction_pattern_width):
    return "cspi_synthetic_dataset_double-hit_diffraction_patterns_{}_uniform_quat_dataset-size={}_diffraction-pattern-shape={}x{}.hdf5".format(dataset_name, dataset_size, diffraction_pattern_height, diffraction_pattern_width)

if __name__ == '__main__':
    main()
