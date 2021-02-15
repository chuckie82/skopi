import h5py
import numpy as np
import os
from skopi.ff_waaskirf_database import load_waaskirf_database, load_cromermann_database

xp = np
if os.environ.get('USE_CUPY') == '1':
    try:
        import cupy as xp
    except ImportError:
        pass
# Ensure arrays are cupy arrays.
asnumpy = np.asarray if xp is np else xp.asnumpy


def deprecation_message(message):
    """Print a deprecation message.

    This function can be used to more easily locate deprecated areas
    in the code.
    """
    print("Deprecation warning: " + message)


def deprecated(reason):
    """Decorator to deprecate a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            deprecation_message(reason)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def prep_h5(output_name):
    """
    Create output file, prepare top level groups, write metadata.

    :param output_name: The output file name.
    :return: None
    """

    with h5py.File(output_name, 'w') as f:
        # Generate top level groups
        f.create_group('data')
        f.create_group('params')
        f.create_group('misc')
        f.create_group('info')

        # Write metadata
        # Package format version
        f.create_dataset('info/package_version', data=np.string_('PySingFEL v0.3.0'))
        # Contact
        f.create_dataset('info/contact', data=np.string_('Carsten Fortmann-Grote <carsten.grote@xfel.eu> for Simex'
                                                         'Haoyuan Li <hyli16@stanford.edu> for PySingFEL'))
        # Data Description
        f.create_dataset('info/data_description',
                         data=np.string_('This dataset contains diffraction patterns generated using SingFEL.'))
        # Method Description
        f.create_dataset('info/method_description', data=np.string_(
            'Form factors of the radiation damaged molecules are calculated in ' +
            'time slices. At each time slice, the coherent scattering is calculated' +
            ' and incoherently added to the final diffraction pattern (/data/nnnnnnn/diffr). ' +
            'Finally, Poissonian noise is added to the diffraction pattern (/data/nnnnnnn/data).'))
        # Data format version
        f.create_dataset('version', data=np.string_('0.2'))


def save_as_diffr_outfile(output_name, input_name, counter, detector_counts, detector_intensity, quaternion, det, beam):
    """
    Save simulation results as new dataset in to the h5py file prepared before.

    :param output_name:
    :param input_name:
    :param counter:
    :param detector_counts:
    :param detector_intensity: The detector intensity
    :param quaternion: The quaternion for each pattern.
    :param det: The detector object
    :param beam: The beam object
    :return:
    """

    with h5py.File(output_name, 'a') as f:
        group_name = '/data/' + '{0:07}'.format(counter + 1) + '/'
        f.create_dataset(group_name + 'data', data=detector_counts)
        f.create_dataset(group_name + 'diffr', data=detector_intensity)
        f.create_dataset(group_name + 'angle', data=quaternion)

        # Link history from input pmi file into output diffr file
        group_name_history = group_name + 'history/parent/detail/'
        f[group_name_history + 'data'] = h5py.ExternalLink(input_name, 'data')
        f[group_name_history + 'info'] = h5py.ExternalLink(input_name, 'info')
        f[group_name_history + 'misc'] = h5py.ExternalLink(input_name, 'misc')
        f[group_name_history + 'params'] = h5py.ExternalLink(input_name, 'params')
        f[group_name_history + 'version'] = h5py.ExternalLink(input_name, 'version')
        f[group_name + '/history/parent/parent'] = h5py.ExternalLink(input_name, 'history/parent')

        # Parameters
        if 'geom' not in f['params'].keys() and 'beam' not in f['params'].keys():
            # Geometry
            f.create_dataset('params/geom/detectorDist', data=det.distance)
            f.create_dataset('params/geom/pixelWidth', data=det.pixel_width[0, 0, 0])
            f.create_dataset('params/geom/pixelHeight', data=det.pixel_width[0, 0, 0])
            f.create_dataset('params/geom/mask', data=np.ones((det.detector_pixel_num_x,
                                                               det.detector_pixel_num_x)))
            f.create_dataset('params/beam/focusArea', data=beam.get_focus_area())

            # Photons
            f.create_dataset('params/beam/photonEnergy', data=beam.get_photon_energy())


def read_geomfile(fname):
    """
    Parse the .geom file to initialize the user defined detector.
    :param fname: The .geom file
    :return: A dict object containing the information of this configuration file.
    """
    # geometry dictionary contains the parameters used to initialize the detector
    geom = {}
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            if line[0] != '#' and line[0] != ';' and len(line) > 1:
                tmp = line.replace('=', ' ').split()
                if "geom" or "panel" in tmp[0]:
                    if tmp[0].split("/")[1] == 'd':
                        geom.update({'distance': float(tmp[1])})
                    if tmp[0].split("/")[1] == 'pix_width':
                        geom.update({'pixel size x': float(tmp[1])})
                        geom.update({'pixel size y': float(tmp[1])})
                    if tmp[0].split("/")[1] == 'px':
                        geom.update({'pixel number x': int(tmp[1])})
                        geom.update({'pixel number y': int(tmp[1])})
            elif line[0] == '#' or line[0] == ';':
                geom.update({'panel number': int(line.strip()[-1])})

    return geom


# Read pdb file and return atom position and type
def symmpdb(fname, ff='WK', filter_occupancy=False):
    """
    Parse the pdb file. This function can handle the REMARK 350 correctly.
    :param fname: The address of the pdb file.
    :param filter_occupancy: Boolean.
        If False, take all atoms.
        If True, only take atoms with > 50% occupancy.
    :return: Numpy array containing the type and position of each atom in the pdb file.
    """

    atom_types = {'H': 1, 'HE': 2, 'C': 6, 'N1+': 6, 'N': 7, 'O': 8, 'O1-': 9, 'P': 15, 'S': 16, 'CL': 18, 'FE': 26}

    fin = open(fname, 'r')

    atoms_dict = {}  # dict to save atom positions and chain id
    sym_dict = {}  # dict to save the symmetry rotations and chain id
    trans_dict = {}  # dict to save the symmetry translations and chain id
    atom_count = 0
    line = fin.readline()
    list_dict = []  # dict for x,y,z,atomtype,atom_symbol,atom_variant,residue
    if ff == 'WK':
        dbase = load_waaskirf_database()
    elif ff == 'CM':
        dbase = load_cromermann_database()
    elif ff == 'SFF':
        dbase = load_waaskirf_database()
    else:
        raise ValueError("Undefined form factor type")

    list1 = [atomType[0] for atomType in dbase]
    # list2 = [charges[0] for charges in dbase]
    while line:
        # read atom coordinates
        if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
            atom_count += 1
            chain_id = line[21]
            if chain_id not in atoms_dict.keys():
                atoms_dict[chain_id] = []
            # occupany > 50 % || one of either if occupany = 50 %
            if (not filter_occupancy or
                float(line[56:60]) > 0.5 or
                (float(line[56:60]) == 0.5 and line[16] != 'B')):

                # [x, y, z, atomtype, charge]
                # Notice that here, one has set the default charge to be 0
                tmp = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()), 0, 0]
                atom_symbol = line[76:78].strip()
                atom_variant = line[13:16].strip()
                residue = line[17:21].strip()

                list_dict.append([tmp[0],tmp[1],tmp[2],tmp[3] ,atom_symbol,atom_variant,residue]) # Get the atom type

                if line[76:78].strip() in atom_types.keys():
                    tmp[3] = atom_types[line[76:78].strip()]
                    idxs = [i for i in range(len(list1)) if list1[i] == tmp[3]]
                    dbase_charges = []
                    for j in idxs:
                        np.append(dbase_charges, dbase[j][1])

                    # Get charge info
                    charge = line[78:80].strip()  # charge info, should be in the form of '2+' or '1-' if not blank
                    if len(charge) != 0:
                        if len(charge) != 2:
                            print('Could not interpret the charge information!\n', line)
                        else:
                            charge = int(charge[1] + charge[0])  # swap the order to be '+2' or '-1' and convert to int
                            if charge in dbase_charges:
                                tmp[4] = charge
                    atoms_dict[chain_id].append(tmp)

                    """if test <= 10:
                        print tmp
                        test+=1"""

                else:
                    print('Unknown element or wrong line: \n', line)

        # read symmetry transformations
        flag1 = 'REMARK 350 APPLY THE FOLLOWING TO CHAINS: '
        flag2 = 'REMARK 350                    AND CHAINS: '
        if line.startswith(flag1):
            line = line.strip()
            chain_ids = line.replace(flag1, '').replace(',', '').split()
            line = fin.readline().strip()
            while line.startswith(flag2):
                chain_ids += line.replace(flag2, '').replace(',', '').split()
                line = fin.readline().strip()
            sys_tmp = []
            trans_tmp = []
            while line[13:18] == 'BIOMT':
                sys_tmp.append([float(line[24:33]), float(line[34:43]), float(line[44:53])])
                trans_tmp.append(float(line[58:68]))
                line = fin.readline().strip()
            sym_dict[tuple(chain_ids)] = np.asarray(sys_tmp)  # cannot use list as dict keys, but tuple works
            trans_dict[tuple(chain_ids)] = np.asarray(trans_tmp)
            # print "find transformation"

            continue

        line = fin.readline()

    fin.close()


    # convert atom positions in numpy array
    for chain_id in atoms_dict.keys():
        atoms_dict[chain_id] = np.asarray(atoms_dict[chain_id])


    # To define a fake atom to initialize the variable
    # When return, this atom is not returned
    atoms = np.zeros((1, 5))

    ##################################################################################################################
    # if no REMARK 350 provided, then save atoms_dict in atoms directly
    if not sym_dict.keys():
        print("no symmetry REMARK 350 found")
        for chain_id in atoms_dict.keys():
            atoms = np.vstack((atoms, atoms_dict[chain_id]))

        x_max = np.max(atoms[:, 0])
        x_min = np.min(atoms[:, 0])
        y_max = np.max(atoms[:, 1])
        y_min = np.min(atoms[:, 1])
        z_max = np.max(atoms[:, 2])
        z_min = np.min(atoms[:, 2])

        # symmetrize atom coordinates
        atoms[:, 0] = atoms[:, 0] - (x_max + x_min) / 2
        atoms[:, 1] = atoms[:, 1] - (y_max + y_min) / 2
        atoms[:, 2] = atoms[:, 2] - (z_max + z_min) / 2

        # Delete the first fake atom
        atom_info = atoms[1:, :]
        # sort based on atomtype and charge
        return atom_info[np.lexsort((atoms[1:, 4].astype(int), atoms[1:, 3].astype(int)))],list_dict

    ##################################################################################################################
    # Deal with the case where we have remark 350
    for chain_ids in sym_dict.keys():
        atoms_array = []
        for chain_id in chain_ids:
            if len(atoms_array) == 0:
                atoms_array = atoms_dict[chain_id]
            else:
                atoms_array = np.vstack((atoms_array, atoms_dict[chain_id]))

        atoms_array_tmp = np.zeros_like(atoms_array)
        atoms_array_tmp[:, :] = atoms_array[:, :]
        sym_array = sym_dict[chain_ids]
        trans_array = trans_dict[chain_ids]
        for i in range(int(len(sym_array) / 3)):
            sym_op = sym_array[3 * i:3 * (i + 1), :]
            trans = trans_array[3 * i:3 * (i + 1)]
            atoms_array_tmp[:, 0:3] = np.dot(atoms_array[:, 0:3], sym_op.T) + trans[np.newaxis, :]
            atoms = np.concatenate((atoms, atoms_array_tmp), axis=0)

    x_max = np.max(atoms[:, 0])
    x_min = np.min(atoms[:, 0])
    y_max = np.max(atoms[:, 1])
    y_min = np.min(atoms[:, 1])
    z_max = np.max(atoms[:, 2])
    z_min = np.min(atoms[:, 2])

    # symmetrize atom coordinates
    atoms[:, 0] = atoms[:, 0] - (x_max + x_min) / 2
    atoms[:, 1] = atoms[:, 1] - (y_max + y_min) / 2
    atoms[:, 2] = atoms[:, 2] - (z_max + z_min) / 2

    # Delete the first fake atom
    atom_info = atoms[1:, :]
    # sort based on atomtype and charge
    return atom_info[np.lexsort((atom_info[:, 4].astype(int), atom_info[:, 3].astype(int)))],list_dict
    # return atom_info, sym_dict, atoms_array
