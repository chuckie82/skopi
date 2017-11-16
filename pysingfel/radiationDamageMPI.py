import argparse
import sys
import time

from mpi4py import MPI

from pysingfel.radiationDamage import *


def main():
    """
    Main function to implement the master-slave model for parallel execution.
    """
    # Delete the first argument from the command line, which is the file name.
    del sys.argv[0]
    # Parse the input command line argumment to get parameters for simulation.
    parameters = parse_input(sys.argv)

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
        print 'Finished: ', end - start, ' seconds.'


def master_diffract(comm, parameters):
    pmiStartID = int(parameters['pmiStartID'])
    pmiEndID = int(parameters['pmiEndID'])
    numDP = int(parameters['numDP'])

    # Number of processes
    numProcesses = comm.Get_size()
    ntasks = (pmiEndID - pmiStartID + 1) * numDP

    if numProcesses == 1:
        rotationAxis = parameters['rotationAxis']
        uniformRotation = parameters['uniformRotation']
        myQuaternions = generateRotations(uniformRotation, rotationAxis, ntasks)
        outputName = parameters['outputDir'] + '/diffr_out_0000001.h5'
        if os.path.exists(outputName):
            os.remove(outputName)
        prepH5(outputName)
        for ntask in range(ntasks):
            MakeOneDiffr(myQuaternions, ntask, parameters, outputName)
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
    for process in range(1, numProcesses):
        comm.send(-1, dest=process)


def slave_diffract(comm, parameters):
    pmiStartID = int(parameters['pmiStartID'])
    pmiEndID = int(parameters['pmiEndID'])
    numDP = int(parameters['numDP'])
    ntasks = (pmiEndID - pmiStartID + 1) * numDP
    rotationAxis = parameters['rotationAxis']
    uniformRotation = parameters['uniformRotation']
    myQuaternions = generateRotations(uniformRotation, rotationAxis, ntasks)

    # Setup output file
    outputName = parameters['outputDir'] + '/diffr_out_' + '{0:07}'.format(comm.Get_rank()) + '.h5'
    if os.path.exists(outputName):
        os.remove(outputName)
    prepH5(outputName)

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
        MakeOneDiffr(myQuaternions, counter, parameters, outputName)
        # Show master we're ready for another task
        comm.send(counter, dest=0, tag=1)


def parse_input(args):
    """
    Parse the input command arguments and return a dict containing all simulation parameters.
    """
    def ParseBoolean(b):
        # Handle different possible Boolean types.
        if b is None:
            return b
        if b == False or b == True:
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

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', help='Input directory for finding /pmi and /diffr')
    parser.add_argument('--outputDir', help='Output directory for saving diffraction')
    parser.add_argument('--beamFile', help='Beam file defining X-ray beam')
    parser.add_argument('--geomFile', help='Geometry file defining diffraction geometry')
    parser.add_argument('--configFile', help='Absolute path to the config file')
    parser.add_argument('--rotationAxis', default='xyz', help='Euler rotation convention')
    parser.add_argument('--uniformRotation', type=ParseBoolean,
                        help='If 1, rotates the sample uniformly in SO(3),\
                                if 0 random orientation in SO(3),\
                                if None (omitted): no orientation.')
    parser.add_argument('--calculateCompton', type=ParseBoolean, default=False,
                        help='If 1, includes Compton scattering in the diffraction pattern')
    parser.add_argument('--sliceInterval', type=int, help='Calculates photon field at every slice interval')
    parser.add_argument('--numSlices', type=int,
                        help='Number of time-slices to use from Photon Matter Interaction (PMI) file')
    parser.add_argument('--pmiStartID', type=int, help='First Photon Matter Interaction (PMI) file ID to use')
    parser.add_argument('--pmiEndID', type=int, help='Last Photon Matter Interaction (PMI) file ID to use')
    parser.add_argument('--numDP', type=int, help='Number of diffraction patterns per PMI file')
    parser.add_argument('--prepHDF5File', help='Absolute path to the prepHDF5.py script')

    # convert argparse to dict
    return vars(parser.parse_args(args))

if __name__ == '__main__':
    main()
