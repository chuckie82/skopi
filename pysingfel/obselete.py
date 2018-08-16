def quaternion2rot3D(quaternion):
    """
    Convert quaternion to a rotation matrix in 3D.
    Use zyz convention after Heymann (2005)
    """
    theta, axis = quaternion2AngleAxis(quaternion)
    return angleAxis2rot3D(axis, theta)


'''
def symmpdb_backup(fname):
    """
    Read REMARK 350 BIOMT from pdb file, which specify the necessary transformation to get the full protein structure.
    Return the atom position as well as atom type in numpy arrays.
    """
    AtomTypes = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16}

    fin = open(fname, 'r')

    atoms_dict = {}  # dict to save atom positions and chain id
    sym_dict = {}  # dict to save the symmetry rotations and chain id
    trans_dict = {}  # dict to save the symmetry translations and chain id
    atom_count = 0
    line = fin.readline()
    while line:
        # read atom coordinates
        if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
            atom_count += 1
            chainID = line[21]
            if chainID not in atoms_dict.keys():
                atoms_dict[chainID] = []
            # occupany > 50 % || one of either if occupany = 50 %
            if (float(line[56:60]) > 0.5) or (float(line[56:60]) == 0.5 and line[16] != 'B'):
                # [x, y, z, atomtype, charge]
                tmp = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()), 0, 0]
                if line[76:78].strip() in AtomTypes.keys():
                    tmp[3] = AtomTypes[line[76:78].strip()]
                    charge = line[78:80].strip()  # charge info, should be in the form of '2+' or '1-' if not blank
                    if len(charge) is not 0:
                        if len(charge) is not 2:
                            print 'Could not interpret the charge information!\n', line
                        else:
                            charge = int(charge[1] + charge[0])  # swap the order to be '+2' or '-1' and convert to int
                            tmp[4] = charge
                    atoms_dict[chainID].append(tmp)
                else:
                    print 'Unknown element or wrong line: \n', line

        # read symmetry transformations
        flag1 = 'REMARK 350 APPLY THE FOLLOWING TO CHAINS: '
        flag2 = 'REMARK 350                    AND CHAINS: '
        if line.startswith(flag1):
            line = line.strip()
            chainIDs = line.replace(flag1, '').replace(',', '').split()
            line = fin.readline().strip()
            while line.startswith(flag2):
                chainIDs += line.replace(flag2, '').replace(',', '').split()
                line = fin.readline().strip()
            sys_tmp = []
            trans_tmp = []
            while line[13:18] == 'BIOMT':
                sys_tmp.append([float(line[24:33]), float(line[34:43]), float(line[44:53])])
                trans_tmp.append(float(line[58:68]))
                line = fin.readline().strip()
            sym_dict[tuple(chainIDs)] = np.asarray(sys_tmp)  # cannot use list as dict keys, but tuple works
            trans_dict[tuple(chainIDs)] = np.asarray(trans_tmp)
            #print "find transformation"

            continue

        line = fin.readline()

    fin.close()

    # convert atom positions in numpy array
    for chainID in atoms_dict.keys():
        atoms_dict[chainID] = np.asarray(atoms_dict[chainID])

    atoms = []

    for chainIDs in sym_dict.keys():
        atoms_array = []
        for chainID in chainIDs:
            if len(atoms_array) == 0:
                atoms_array = atoms_dict[chainID]
            else:
                atoms_array = np.vstack((atoms_array, atoms_dict[chainID]))
        sym_array = sym_dict[chainIDs]
        trans_array = trans_dict[chainIDs]
        for i in range(int(len(sym_array) / 3)):
            sym_op = sym_array[3 * i:3 * (i + 1), :]
            trans = trans_array[3 * i:3 * (i + 1)]
            atoms_array[:, 0:3] = np.dot(atoms_array[:, 0:3], sym_op.T) + np.tile(trans, (len(atoms_array), 1))
            if len(atoms) == 0:
                atoms = atoms_array
            else:
                atoms = np.vstack((atoms, atoms_array))

    # if no REMARK 350 provided, then save atoms_dict in atoms directly
    if not sym_dict.keys():
        print "no 350 found"
        for chainID in atoms_dict.keys():
            if len(atoms) == 0:
                atoms = atoms_dict[chainID]
            else:
                atoms = np.vstack((atoms, atoms_dict[chainID]))

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

    # sort based on atomtype and charge
    return atoms[np.lexsort((atoms[:,4].astype(int), atoms[:,3].astype(int)))]
'''


def set_detector_parameters_and_initialize(self, oname=None, oindex=0,
                                           x0=0, y0=0, z0=0, rot_z=0, rot_y=0, rot_x=0,
                                           tilt_z=0, tilt_y=0, tilt_x=0):
    self.geometry.set_detector_parameters(oname, oindex, x0, y0, z0,
                                          rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)
    self._initialize()


def set_detector_parameters(self, oname=None, oindex=0,
                            x0=0, y0=0, z0=0, rot_z=0, rot_y=0, rot_x=0,
                            tilt_z=0, tilt_y=0, tilt_x=0):
    self.geometry.set_detector_parameters(oname, oindex, x0, y0, z0,
                                          rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)


###########################################################################
## The following funcitons use tensorflow to do calculation
###########################################################################

'''
import tensorflow as tf

#This method also works. But the efficiency is low. So currently we are using the numba.cuda rather than
#the tensorflow. But if you are interested, you can also check this method.

def calculate_diffraction_volume_gpu(particle, q_space, q_position):
    f_hkl = calculate_atomicFactor_3d(particle, q_space)
    split_index = particle.SplitIdx[:]
    atomPos = particle.atomPos[:]
    size_x, size_y, size_z = q_space.shape
    pattern = np.zeros_like(q_space)

    atomNumber = split_index[-1]

    atomSpecies = np.zeros(split_index[-1], dtype=int)

    for i in range(len(split_index)-1):
        atomSpecies[split_index[i]:split_index[i+1]] = i

    form_factor = tf.constant(value= f_hkl, dtype= tf.complex64)
    split_index = tf.constant(value= split_index)
    atom_position = tf.constant(value= atomPos, dtype= tf.complex64)
    reciprocal_position = tf.constant(value= q_position, dtype= tf.complex64)
    atom_species = tf.constant(value = atomSpecies)
    atom_number = tf.constant(value = atomNumber)

    pattern_3d =  tf.Variable(tf.zeros([size_x, size_y, size_z], dtype=tf.complex64),
                                  dtype= tf.complex64)

    number = tf.Variable(0 ,dtype= tf.int64)

    def condition(number, pattern_3d):
        return number < 10
    def body(number, pattern_3d):
        pattern_3d = tf.add(pattern_3d , tf.multiply(form_factor[:,:,:,atom_species[number]],
                                     tf.exp(1j* 
                                           tf.einsum('ijkl,l->ijk', 
                                                     reciprocal_position, atom_position[number]))))
        number += 1
        return number , pattern_3d

    init_op = tf.global_variables_initializer()
    index,pattern_holder = tf.while_loop(condition, body, [number , pattern_3d], 
                              shape_invariants=[number.get_shape(), pattern_3d.get_shape()],
                                        parallel_iterations=1000)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      # Run the init operation.
        sess.run(init_op)
        number_of_iteration, pattern = sess.run([index, pattern_holder ])

    return number_of_iteration, np.abs(pattern) ** 2
'''