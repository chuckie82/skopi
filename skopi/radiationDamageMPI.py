import argparse, os

from mpi4py import MPI

from skopi.radiationDamage import *
import skopi.util as su


def main():
    """
    Main function to implement the master-slave model for parallel execution.

    :return:
    """

    parameters = parse_input()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Initialize time
    start = time.time()
    if rank == 0:
        master_diffract(comm, parameters)
    else:
        slave_diffract(comm, parameters)

    comm.Barrier()  # Barrier synchronization

    if rank == 0:
        end = time.time()
        print('Finished: ', end - start, ' seconds.')


def master_diffract(comm, parameters):
    """
    Master node. Get the diffraction patterns with mpi

    :param comm: MPI comm
    :param parameters: dictionary of command line arguments
    :return:
    """
    pmi_start_id = int(parameters['pmiStartID'])
    pmi_end_id = int(parameters['pmiEndID'])
    num_dp = int(parameters['numDP'])

    # Number of processes
    num_process = comm.Get_size()
    ntasks = (pmi_end_id - pmi_start_id + 1) * num_dp

    if num_process == 1:
        rotation_axis = parameters['rotationAxis']
        uniform_rotation = parameters['uniformRotation']
        my_quaternion = generate_rotations(uniform_rotation, rotation_axis, ntasks)
        output_name = parameters['outputDir'] + '/diffr_out_0000001.h5'
        if os.path.exists(output_name):
            os.remove(output_name)
        su.prep_h5(output_name)
        for ntask in range(ntasks):
            make_one_diffr(my_quaternion, ntask, parameters, output_name)
    else:
        for ntask in range(ntasks):
            status = MPI.Status()
            # Waiting for messages from slave
            # Successful message reciving means slave is ready for simulation
            comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
            rnk = status.Get_source()
            # Trigger calculation on slave.
            comm.send(ntask, dest=rnk)

    # Final send: stop all processes from waiting for tasks
    for process in range(1, num_process):
        comm.send(-1, dest=process)


def slave_diffract(comm, parameters):
    """
    Slave node. Get the diffraction patterns with mpi

    :param comm: MPI comm
    :param parameters: dictionary of command line arguments
    :return:
    """
    pmi_start_id = int(parameters['pmiStartID'])
    pmi_end_id = int(parameters['pmiEndID'])
    num_dp = int(parameters['numDP'])
    ntasks = (pmi_end_id - pmi_start_id + 1) * num_dp
    rotation_axis = parameters['rotationAxis']
    uniform_rotation = parameters['uniformRotation']
    my_quaternion = generate_rotations(uniform_rotation, rotation_axis, ntasks)

    # Setup output file
    output_name = parameters['outputDir'] + '/diffr_out_' + '{0:07}'.format(comm.Get_rank()) + '.h5'
    if os.path.exists(output_name):
        os.remove(output_name)
    su.prep_h5(output_name)

    # Init a local counter
    counter = 0
    # Wave to master, we're good to go.
    comm.send(counter, dest=0, tag=1)
    # Start event loop and generate the diffraction images.
    while True:
        counter = comm.recv(source=0)
        if counter < 0:
            # end of simulation
            return None
        make_one_diffr(my_quaternion, counter, parameters, output_name)
        # Show master we're ready for another task
        comm.send(counter, dest=0, tag=1)


def parse_input():
    """
    Parse the input command arguments and return a dict containing all simulation parameters.

    :return parameters: dictionary of command-line arguments
    """

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', help='Input directory for finding /pmi and /diffr')
    parser.add_argument('--outputDir', help='Output directory for saving diffraction')
    parser.add_argument('--beamFile', help='Beam file defining X-ray beam')
    parser.add_argument('--geomFile', help='Geometry file defining diffraction geometry')
    parser.add_argument('--rotationAxis', default='xyz', help='Preferred axis of rotation or xyz if none')
    parser.add_argument('--uniformRotation', type=parse_boolean,
                        help='If 1, rotates the sample uniformly in SO(3),\
                                if 0 random orientation in SO(3),\
                                if None (omitted): no orientation.')
    parser.add_argument('--calculateCompton', type=parse_boolean, default=False,
                        help='If 1, includes Compton scattering in the diffraction pattern')
    parser.add_argument('--sliceInterval', type=int, help='Calculates photon field at every slice interval')
    parser.add_argument('--numSlices', type=int,
                        help='Number of time-slices to use from Photon Matter Interaction (PMI) file')
    parser.add_argument('--pmiStartID', type=int, help='First Photon Matter Interaction (PMI) file ID to use')
    parser.add_argument('--pmiEndID', type=int, help='Last Photon Matter Interaction (PMI) file ID to use')
    parser.add_argument('--numDP', type=int, help='Number of diffraction patterns per PMI file')

    # convert argparse to dict
    return vars(parser.parse_args())


def parse_boolean(b):
    """
    Handle different possible Boolean types.

    :param b:
    :return:
    """
    if b is None:
        return b
    if b is False or b is True:
        return b
    b = b.strip()
    if len(b) < 1:
        raise ValueError('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError('Cannot parse string into boolean.')


if __name__ == '__main__':
    main()
