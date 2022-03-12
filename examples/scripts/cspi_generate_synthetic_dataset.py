import time
import os
import sys

import argparse
import json

import pprint
import tqdm

import numpy as np

from PIL import Image

import h5py as h5

import skopi as sk
from skopi.detector.pnccd import PnccdDetector


"""

Deeban Ramalingam (deebanr@slac.stanford.edu)

This script generates a synthetic dataset of diffraction patterns and their associated orientations for computational Single Particle Imaging (cSPI). This script uses Skopi to simulate an SPI Experiment.

How to run this script:

python cspi_generate_synthetic_dataset.py --config <path to Config file> --dataset <alias for the dataset>

Examples on how to run this script:

1. python cspi_generate_synthetic_dataset.py --config cspi_generate_synthetic_dataset_config.json --dataset 3iyf

2. python cspi_generate_synthetic_dataset.py --config cspi_generate_synthetic_dataset_config.json --dataset two-atoms-100

Tips on using this script:

1. For an example config file, look at: skopi/examples/scripts/cspi_generate_synthetic_dataset_config.json

2. Use the previously defined datasets in this file to add or modify an existing experiment of your choice.

If you wish to use the Latent Space Visualizer to visualize the synthetic dataset, make the image output directory accessible to JupyterHub on PSWWW after running the script.

Examples on how to make the image output directory accessible to the Latent Space Visualizer:

1. ln -s /reg/data/ana03/scratch/deebanr/3iyf /reg/neh/home/deebanr/3iyf

2. ln -s /reg/data/ana03/scratch/deebanr/two-atoms-100 /reg/neh/home/deebanr/two-atoms-100

Tips on creating file system links with the ln command from the Terminal:

https://www.linode.com/docs/tools-reference/tools/create-file-system-links-with-ln/#use-cases-for-symbolic-links

"""


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
    
    # Get the input and output dataset parameters
    pdb_file = dataset_params['pdb']
    beam_file = dataset_params['beam']
    beam_fluence_increase_factor = dataset_params['beamFluenceIncreaseFactor']
    geom_file = dataset_params['geom']
    dataset_size = dataset_params['numPatterns']
    img_dir = dataset_params['imgDir']
    output_dir = dataset_params['outDir']
    
    # PDB
    print("Load PDB: {}".format(pdb_file))
    particle = sk.Particle()
    particle.read_pdb(pdb_file, ff='WK')
    atomic_coordinates = particle.atom_pos
    
    # Beam parameters
    print("Load beam parameters: {}".format(pdb_file))
    beam = sk.Beam(beam_file)

    # Increase the beam fluence
    if not np.isclose(beam_fluence_increase_factor, 1.0):
        print('BEFORE: # of photons per pulse {}'.format(beam.get_photons_per_pulse()))
        print('>>> Increasing the number of photons per pulse by a factor {}'.format(beam_fluence_increase_factor))
        beam.set_photons_per_pulse(beam_fluence_increase_factor * beam.get_photons_per_pulse())
        print('AFTER : # of photons per pulse {}'.format(beam.get_photons_per_pulse()))

    # Geometry of detector
    print("Load detector geometry: {}".format(geom_file))
    det = PnccdDetector(geom=geom_file, beam=beam)

    
    # Simulate the SPI Experiment
    print("Calculating diffraction volume")
    
    tic = time.time()
    
    experiment = sk.SPIExperiment(det, beam, particle)
    
    toc = time.time()

    print("It takes {:.2f} seconds to finish the calculation.".format(toc-tic))

    # Generate random orientations
    print("Generating random orientations as uniform quaternions")
    orientations = sk.get_uniform_quat(dataset_size, True)
    
    # Get diffraction pattern shape
    diffraction_pattern_height = det.detector_pixel_num_x.item()
    diffraction_pattern_width = det.detector_pixel_num_y.item()

    # Use orientations to generate diffraction patterns
    print("Using orientations to generate diffraction patterns")
    diffraction_patterns = np.zeros((dataset_size, diffraction_pattern_height, diffraction_pattern_width))
    experiment.set_orientations(orientations)
    
    tic = time.time()
    
    for data_index in tqdm.tqdm(range(dataset_size)):
        diffraction_pattern = experiment.generate_image()
        diffraction_patterns[data_index] = diffraction_pattern
        save_diffraction_pattern_as_image(data_index, img_dir, diffraction_pattern)
    
    toc = time.time()
    
    print("It takes {:.2f} seconds to generate the diffraction patterns.".format(toc-tic))
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        print("Creating output directory: {}".format(output_dir))
        os.makedirs(output_dir)

    # Define path to output HDF5 file
    output_file = get_output_file_name(dataset_name, dataset_size, diffraction_pattern_height, diffraction_pattern_width)    
    cspi_synthetic_dataset_file = os.path.join(output_dir, output_file)
    print("Saving dataset to: {}".format(cspi_synthetic_dataset_file))

    # Define dataset names for HDF5 file
    diffraction_patterns_dataset_name = "diffraction_patterns"
    orientations_dataset_name = "orientations"
    atomic_coordinates_dataset_name = "atomic_coordinates"

    # Create and write datasets to HDF5 file
    with h5.File(cspi_synthetic_dataset_file, "w") as cspi_synthetic_dataset_file_handle:
        dset_diffraction_patterns = cspi_synthetic_dataset_file_handle.create_dataset(diffraction_patterns_dataset_name, diffraction_patterns.shape, dtype='f')
        dset_diffraction_patterns[...] = diffraction_patterns
        dset_orientations = cspi_synthetic_dataset_file_handle.create_dataset(orientations_dataset_name, orientations.shape, dtype='f')
        dset_orientations[...] = orientations
        dset_atomic_coordinates = cspi_synthetic_dataset_file_handle.create_dataset(atomic_coordinates_dataset_name, atomic_coordinates.shape, dtype='f')
        dset_atomic_coordinates[...] = atomic_coordinates
        
    # Load datasets from HDF5 file to verify write
    with h5.File(cspi_synthetic_dataset_file, "r") as cspi_synthetic_dataset_file_handle:
        print("cspi_synthetic_dataset_file keys:", list(cspi_synthetic_dataset_file_handle.keys()))
        print(cspi_synthetic_dataset_file_handle[diffraction_patterns_dataset_name])
        print(cspi_synthetic_dataset_file_handle[orientations_dataset_name])
        print(cspi_synthetic_dataset_file_handle[atomic_coordinates_dataset_name])
        diffraction_patterns = cspi_synthetic_dataset_file_handle[diffraction_patterns_dataset_name][:]
        
    # compute statistics
    print("Diffraction pattern statistics:")
    diffraction_pattern_statistics = {
        'min': diffraction_patterns.min(),
        'max': diffraction_patterns.max(),
        'mean': diffraction_patterns.mean()
    }
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(diffraction_pattern_statistics)

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: two-atoms, 3iyf')

    # convert argparse to dict
    return vars(parse.parse_args(args))

def save_diffraction_pattern_as_image(data_index, img_dir, diffraction_pattern):
    # Create image output directory if it does not exist
    if not os.path.exists(img_dir):
        print("Creating image output directory: {}".format(img_dir))
        os.makedirs(img_dir)

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
    return "cspi_synthetic_dataset_diffraction_patterns_{}_uniform_quat_dataset-size={}_diffraction-pattern-shape={}x{}.hdf5".format(dataset_name, dataset_size, diffraction_pattern_height, diffraction_pattern_width)

if __name__ == '__main__':
    main()
