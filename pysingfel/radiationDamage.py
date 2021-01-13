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
        if rotation_axis == 'y' or rotation_axis == 'z':
            return points_on_1sphere(num_quaternions, rotation_axis)
        elif rotation_axis == 'xyz':
            return points_on_3sphere(num_quaternions)
    else:
        # Case non-uniform:
        quaternions = get_random_quat(num_quaternions)
        return quaternions


def set_energy_from_file(fname, beam):
    """
    Set photon energy from pmi file.

    :param fname: pmi file
    :param beam: beam object
    """

    with h5py.File(fname, 'r') as f:
        photon_energy = f.get('/history/parent/detail/params/photonEnergy').value
    beam.set_photon_energy(photon_energy)


def set_focus_from_file(fname, beam):
    """
    Set beam focus from pmi file.

    :param fname: pmi file
    :param beam: beam object
    """

    with h5py.File(fname, 'r') as f:
        focus_xfwhm = f.get('/history/parent/detail/misc/xFWHM').value
        focus_yfwhm = f.get('/history/parent/detail/misc/yFWHM').value
    beam.set_focus(focus_xfwhm, focus_yfwhm, shape='ellipse')


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
            n_phot += f.get(datasetname).value
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

    # Set up beam and detector from file
    beam = Beam(beamfile)

    # If not given, read from pmi file later
    given_fluence = False
    if beam.get_photons_per_pulse() > 0:
        given_fluence = True
    given_photon_energy = False
    if beam.get_photon_energy() > 0:
        given_photon_energy = True
    given_focus_radius = False
    if (beam.get_focus()[0] > 0) and (beam.get_focus()[1] > 0):
        given_focus_radius = True

    # Setup quaternion.
    quaternion = myquaternions[counter, :]

    # Input file
    input_name = input_dir + '/pmi_out_' + '{0:07}'.format(pmi_id) + '.h5'

    # Set up diffraction geometry
    if not given_photon_energy:
        set_energy_from_file(input_name, beam)
    if not given_focus_radius:
        set_focus_from_file(input_name, beam)

    # Detector geometry
    det = PlainDetector(geom=geomfile, beam=beam)
    px = det.detector_pixel_num_x
    py = det.detector_pixel_num_x

    done = False
    time_slice = 0
    total_phot = 0
    detector_intensity = np.zeros((1, py, px))
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
            set_fluence_from_file(input_name, time_slice, slice_interval, beam)
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
