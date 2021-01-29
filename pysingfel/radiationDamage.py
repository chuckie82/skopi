from pysingfel.geometry import *
from pysingfel.particle import Particle
import h5py
from pysingfel.detector import *
from pysingfel.beam import *
import pysingfel.util as pu
from pysingfel.diffraction import calculate_compton


def generate_rotations(uniform_rotation, rotation_axis, num_quaternions):
    """
    Return quaternions saving the rotations to the particle.

    :param uniform_rotation: Bool, if True distribute points evenly
    :param rotation_axis: rotation axis or 'xyz' if no preferred axis
    :param num_quaternions: number of quaternions to generate
    :return: quaternion list of shape [number of quaternion, 4]
    """

    if not uniform_rotation:
        # No rotation desired, init quaternions as (1,0,0,0)
        quaternions = np.empty((num_quaternions, 4))
        quaternions[:, 0] = 1.
        quaternions[:, 1:] = 0.

        return quaternions

    # Case uniform:
    if uniform_rotation and num_quaternions!=1:
        if rotation_axis == 'x' or rotation_axis == 'y' or rotation_axis == 'z':
            return points_on_1sphere(num_quaternions, rotation_axis)
        elif rotation_axis == 'xyz':
            return points_on_3sphere(num_quaternions)
    else:
        # edge case of one quaternion requested
        quaternions = get_random_quat(num_quaternions)
        return quaternions


def initialize_beam_from_pmi(fname):
    """
    Generate beam object from pmi file, with fluence set to 0.
    :param fname: pmi file
    :return beam: beam object
    """
    params = dict()
    with h5py.File(fname, 'r') as f:
        # Set photon_energy
        if "photon_energy" in f['params'].keys():
            # SimEx PMI output format
            params['photon_energy'] = f.get('/params/photon_energy')[()]
        elif 'xparams' in f['params'].keys():
            lines = [
                l.split(' ')
                for l in f['params/xparams'][()].decode('utf-8').split("\n")
            ]
            xparams = get_dict_from_lines(lines)
            params['photon_energy'] = xparams['EPH']
        else:
            # Legacy support: PMI output before the development of SimEx
            params['photon_energy'] = f.get(
                '/history/parent/detail/params/photonEnergy')[()]

        # Set focus
        if "focus" in f['params'].keys():
            # SimEx PMI output format
            params['focus_x'] = f.get('/params/focus/xFWHM')[()]
            params['focus_y'] = f.get('/params/focus/yFWHM')[()]
        elif 'xparams' in f['params'].keys():
            lines = [
                l.split(' ')
                for l in f['params/xparams'][()].decode('utf-8').split("\n")
            ]
            xparams = get_dict_from_lines(lines)
            diam = xparams['DIAM']
            params['focus_x'] = diam
            params['focus_y'] = diam
        else:
            # Legacy support: PMI output before the development of SimEx
            params['focus_x'] = f.get('/history/parent/detail/misc/xFWHM')[()]
            params['focus_y'] = f.get('/history/parent/detail/misc/yFWHM')[()]

        params['focus_shape'] = 'ellipse'
        params['fluence'] = 0  # placeholder, gets reset at each shot

    beam = Beam(**params)
    return beam


def get_dict_from_lines(reader):
    """ Turn a list of [key, ' ', ..., value] elements into a dict.

    :params reader: An iterable that contains lists of strings in format [key, ' ', ' ', ..., value]
    :type: iterable (list, array, generator).
    :return ret: A dictionary of beam-related parameters.
    """
    # These fields shall be handled as numeric data.
    numeric_keys = [
        'N',
        'Z',
        'DIST',
        'EPH',
        'NPH',
        'DIAM',
        'FLU_MAX',
        'T',
        'T0',
        'R0',
        'DT',
        'STEPS',
        'PROGRESS',
        'RANDSEED',
        'RSTARTE',
    ]
    # Initialize return dictionary.
    ret = dict()

    # Iteratoe through all lines.
    for line in reader:
        # Skip empty lines and comments.
        if line == [] or line == ['']:
            continue
        if line[0][0] == '#':
            continue

        # Get key-value pair (they're separated by random number of whitespaces.
        key, val = line[0], line[-1]

        # Fix numeric data.
        if key in numeric_keys:
            try:
                val = float(val)
            except:
                raise

        # Store on dict.
        ret[key] = val

    # Return finished dict.
    return ret


def set_fluence_from_file(fname, time_slice, slice_interval, beam):
    """
    Set beam fluence from pmi file.

    :param fname: pmi file
    :param time_slice: time of current calculation
    :param slice_interval: interval for calculating photon field
    :param beam: beam
    """

    n_phot = 0
    for i in range(slice_interval):
        with h5py.File(fname, 'r') as f:
            datasetname = '/data/snp_' + '{0:07}'.format(time_slice - i) + '/Nph'
            n_phot += f.get(datasetname)[()]
    beam.set_photons_per_pulse(n_phot)


def make_one_diffr(myquaternions, counter, parameters, output_name):
    """
    Generate one diffraction pattern related to a certain rotation.
    Write results in output hdf5 file.

    :param myquaternions: list of quaternions
    :param counter: index of diffraction pattern to compute
    :param parameters: dictionary of command line arguments
    :param output_name: path to h5 file for saving pattern
    """

    # Get parameters
    consider_compton = parameters['calculateCompton']
    num_dp = int(parameters['numDP'])
    num_slices = int(parameters['numSlices'])
    pmi_start_id = int(parameters['pmiStartID'])
    pmi_id = int(pmi_start_id + counter / num_dp)
    slice_interval = int(parameters['sliceInterval'])
    beamfile = parameters['beamFile']
    geomfile = parameters['geomFile']
    input_dir = parameters['inputDir']

    # Input file
    input_name = input_dir + '/pmi_out_' + '{0:07}'.format(pmi_id) + '.h5'

    # Set up quaternion
    quaternion = myquaternions[counter, :]

    # Set up beam from beam or pmi file
    given_fluence =True
    if beamfile is not None:
        beam = Beam(beamfile)
    else:
        beamfile = input_name
        beam = initialize_beam_from_pmi(beamfile)
        given_fluence = False

    # Detector geometry
    det = PlainDetector(geom=geomfile, beam=beam)
    px = det.detector_pixel_num_x
    py = det.detector_pixel_num_x

    done = False
    time_slice = 0
<<<<<<< HEAD
    detector_intensity = np.zeros((py, px))
||||||| 2e00b2e
    total_phot = 0
    detector_intensity = np.zeros((py, px))
=======
    total_phot = 0
    detector_intensity = np.zeros((1, py, px))
>>>>>>> 6fe3923708e13a92a39187d546e8e026b25f8983
    while not done:
        # set time slice to calculate diffraction pattern
        if time_slice + slice_interval >= num_slices:
            slice_interval = num_slices - time_slice
            done = True
        time_slice += slice_interval

        # load particle information
        datasetname = '/data/snp_' + '{0:07}'.format(time_slice)
        particle = Particle(input_name, datasetname)
        particle.rotate(quaternion)
        if not given_fluence:
            # sum up the photon fluence inside a slice_interval
            set_fluence_from_file(beamfile, time_slice, slice_interval, beam)
        # Coherent contribution
        f_hkl_sq = det.get_pattern_without_corrections(particle)

        # Incoherent contribution
        if consider_compton:
            compton = calculate_compton(particle, det)
        else:
            compton = 0
        photon_field = f_hkl_sq + compton
        detector_intensity += photon_field*beam.get_photons_per_pulse_per_area()
    detector_intensity *= (det.solid_angle_per_pixel *
                           det.polarization_correction) * det.Thomson_factor

    detector_counts = np.random.poisson(detector_intensity)
    pu.save_as_diffr_outfile(output_name, input_name, counter,
                             detector_counts, detector_intensity,
                             quaternion, det, beam)


def diffract(parameters):
    """
    Calculate all the diffraction patterns based on the parameters provided as a dictionary.
    Save all results in one single file. Not used in MPI.

    :param parameters: dictionary of command line arguments
    """

    pmi_start_id = int(parameters['pmiStartID'])
    pmi_end_id = int(parameters['pmiEndID'])
    num_dp = int(parameters['numDP'])
    ntasks = (pmi_end_id - pmi_start_id + 1) * num_dp
    rotation_axis = parameters['rotationAxis']
    uniform_rotation = parameters['uniformRotation']
    myquaternions = generate_rotations(uniform_rotation, rotation_axis, ntasks)
    output_name = parameters['outputDir'] + '/diffr_out_0000001.h5'
    if os.path.exists(output_name):
        os.remove(output_name)
    pu.prep_h5(output_name)
    for ntask in range(ntasks):
        make_one_diffr(myquaternions, ntask, parameters, output_name)
