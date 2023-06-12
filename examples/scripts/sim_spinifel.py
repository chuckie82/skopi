import argparse, time, os
import numpy as np
import skopi as sk
import h5py, sys

"""
Script for generating SPI datasets for reconstruction in spinifel. Either noise-free
diffraction intensities or photons affected by beam jitter, fluence jitter, and/or a
static sloped background on top of Poisson noise can be simulated on either a simple
monolithic square or an LCLS-style detector. The dataset is saved to an h5 file.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Simulate a simple SPI dataset.")
    parser.add_argument('-b', '--beam_file', help='Beam file', required=True, type=str)
    parser.add_argument('-p', '--pdb_file', help='Atomic coordinates file in PDB format', required=True, type=str)
    parser.add_argument('-d', '--det_info', help='Detector info, either (n_pixels, width, distance) for SimpleSquare'+
                        'or (det_type, geom_file, distance) for LCLSDetectors. det_type could be pnccd, for instance',
                        required=True, nargs=3)
    parser.add_argument('-n', '--n_images', help='Number of diffraction images to compute', required=True, type=int)
    parser.add_argument('-q', '--quantize', help='Add Poisson noise to compute photons rather than intensities', action='store_true')
    parser.add_argument('-s', '--increase_factor', help='Scale factor by which to increase beam fluence', required=False, default=1, type=float)
    parser.add_argument('-o', '--output', help='Path to output h5 file', required=False, type=str)
    parser.add_argument('-bj', '--beam_jitter', help='Sigma for Gaussian beam jitter in pixels', required=False, type=float, default=0)
    parser.add_argument('-fj', '--fluence_jitter', help='Sigma for Gaussian jitter as a fraction of total fluence', required=False, type=float, default=0)
    parser.add_argument('-sb', '--sloped', help='Sloped background array in .npy format', required=False, type=str)

    return vars(parser.parse_args())


def setup_experiment(args):
    """
    Set up experiment class.
    
    :param args: dict containing beam, pdb, and detector info
    :return exp: skopi SPIExperiment object
    """
    beam = sk.Beam(args['beam_file'])
    if args['increase_factor'] != 1:
        beam.set_photons_per_pulse(args['increase_factor']*beam.get_photons_per_pulse())
    
    particle = sk.Particle()
    particle.read_pdb(args['pdb_file'], ff='WK')

    if args['det_info'][0].isdigit():
        n_pixels, det_size, det_dist = args['det_info']
        det = sk.SimpleSquareDetector(int(n_pixels), float(det_size), float(det_dist), beam=beam) 
    elif args['det_info'][0] == 'pnccd':
        det = sk.PnccdDetector(geom=args['det_info'][1])
        det.distance = float(args['det_info'][2])
    elif args['det_info'][0] == 'cspad':
        det = sk.CsPadDetector(geom=args['det_info'][1])
        det.distance = float(args['det_info'][2])
    else:
        print("Detector type not recognized. Must be pnccd, cspad, or SimpleSquare.")
    
    exp = sk.SPIExperiment(det, beam, particle)
    exp.set_orientations(sk.get_random_quat(args['n_images']))
    
    return exp


def simulate_writeh5(args):
    """
    Simulate diffraction images and save to h5 file.
    :param args: dictionary of command line input
    """
    print("Simulating diffraction images")
    start_time = time.time()
    command_line = sys.argv

    # set image type
    if args['quantize']:
        itype = 'photons'
    else:
        itype = 'intensities'

    # set up experiment and create h5py file
    exp = setup_experiment(args)
    f = h5py.File(args["output"], "w")

    # store useful experiment arrays
    print("pix_pos: ", type(exp.det.pixel_position_reciprocal))
    f.create_dataset("pixel_position_reciprocal", data=exp.det.pixel_position_reciprocal.get()) # s-vectors in m-1 
    f.create_dataset("volume", data=exp.volumes[0]) # reciprocal space volume, 151 pixels cubed
    f.create_dataset("pixel_index_map", data=exp.det.pixel_index_map.get()) # indexing map for reassembly
    f.create_dataset("orientations", data=np.expand_dims(exp._orientations, axis=1)) # ground truth quaternions
    f.create_dataset("polarization", data=exp.det.polarization_correction.get()) # polarization correction
    f.create_dataset("solid_angle", data=exp.det.solid_angle_per_pixel.get()) # solid angle correction 

    # simulate images and save to h5 file
    imgs = f.create_dataset('intensities', shape=((args['n_images'],) + exp.det.shape))
    for num in range(args['n_images']):
        if itype == 'intensities':
            imgs[num,:,:,:] = exp.generate_image_stack(return_intensities=True, noise=args['noise']).get()
        else:
            imgs[num,:,:,:] = exp.generate_image_stack(return_photons=True, noise=args['noise']).get()

    # save beam offsets, fluences
    f.create_dataset("beam_offsets", data=np.array(exp.beam_displacements)) # beam displacements per shot
    f.create_dataset("fluences", data=np.array(exp.fluences)) # number of photons per shot

    # save useful attributes
    f.attrs['reciprocal_extent'] = np.linalg.norm(exp.det.pixel_position_reciprocal.get(), axis=-1).max() # max |s|
    f.attrs['n_images'] = args['n_images'] # number of simulated shots
    f.attrs['n_pixels_per_image'] = exp.det.pixel_num_total # number of total pixels per image
    f.attrs['det_shape'] = exp.det.shape # detector shape (n_panels, panel_num_x, panel_num_y)
    f.attrs['det_distance'] = float(args['det_info'][2]) # detector distance in meters
    f.attrs['command_line'] = " ".join(sys.argv) # command line arguments
    
    f.close()

    print("Simulated dataset saved to %s" %args['output'])
    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))

    return


def main():

    # gather command line input 
    args = parse_input()

    # assemble noise dictionary
    args['noise'] = dict()
    if args['beam_jitter']!=0:
        args['noise']['beam_offset'] = args['beam_jitter']
    if args['fluence_jitter']!=0:
        args['noise']['fluence_jitter'] = args['fluence_jitter']
    if args['sloped']:
        args['noise']['sloped'] = np.load(args['sloped'])
    print("Types of errors to be simulated:")
    if args['quantize']:
        print("Poisson noise")
    print(args['noise'])

    # simulate images and save
    simulate_writeh5(args)

if __name__ == '__main__':
    main()
