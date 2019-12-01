import h5py
import numpy as np
import sys
from util import symmpdb
#from geometry import quaternion2rot3d, get_random_rotation, get_random_translations
from ff_waaskirf_database import *


class Particle(object):
    """
    Class to hold the particle information
    """

    def __init__(self, *fname):
        """
        Initialize the particle with the .pdb file specified with fname
        :param fname: The frames to read.
        """
        # Atom positions, types and form factor table
        self.atom_pos = None  # atom position -> N x 3 array, sorted based on atom type id
        # Index array saving indices that split atom_pos to get pos for each atom type
        # More specifically, let m = split_idx[i] and n = split_idx[i+1], then
        # atom_pos[m:n] contains all atoms for the ith atom type.

        self.trans = None

        self.split_idx = None
        self.num_atom_types = None  # number of atom types
        self.ff_table = None  # form factor table -> atom_type x qSample
        

        # Scattering
        self.q_sample = None  # q vector sin(theta)/lambda
        self.num_q_samples = None  # number of q samples
        # Compton scattering
        self.compton_q_sample = None  # Compton: q vector sin(theta)/lambda
        self.num_compton_q_samples = 0  # number of Compton q samples
        self.sBound = None  # Compton: static structure factor S(q)
        self.nFree = None  # Compton: number of free electrons
        self.element = None
        self.residue = None
        self.atomic_symbol = None
        self.at = None
        if len(fname) != 0:
            # read from pmi file to get info about radiation damage at a certain time slice
            if len(fname) == 1:
                datasetname = 'data/snp_0000001'  # default dataset name -> set to be initial time
                self.read_h5file(fname[0], datasetname)
            elif len(fname) == 2:
                # both pmi file and the time slice (dataset) are provided
                self.read_h5file(fname[0], fname[1])
            else:
                raise ValueError('Wrong number of parameters to construct the particle object!')
    
    def get_atom_type(self):
        #print("Self_at",self.at)
        return self.at
        
    def get_atom_struct(self):
        return self.atom_struct
    
    def get_atomic_symbol(self):
        return self.atomic_symbol
        
    def get_atomic_variant(self):
        return self.atomic_variant
        
    def get_residue(self):
        return self.residue
        
    # Generate some random rotation in the particle
    def rotate(self, quaternion):
        """
        Rotate the particle with the specified quaternion

        :param quaternion:
        :return: None
        """
        rot3d = quaternion2rot3d(quaternion)
        new_pos = np.dot(self.atom_pos, rot3d.T)
        self.set_atom_pos(new_pos)

    def rotate_randomly(self, axis='y'):
        """
        Rotate randomly.
        :param axis: 'y' for random rotation around y axis.
                    Anything else for a totally random rotation
        :return: None
        """
        rot3d = get_random_rotation(axis)
        new_pos = np.dot(self.atom_pos, rot3d.T)
        self.set_atom_pos(new_pos)
    

    def random_translation_vector(self):#, beam_focus_size):
        """
        Gives a random translation vector
        :param beam_focus_size: Radius within which we want our translation
        :return: translation vector
        """
        #N = len(self.atom_pos)
        #x_trans = beam_focus_size*np.random.uniform(-1, 1)
        #y_trans = beam_focus_size*np.random.uniform(-1, 1)
        #trans = [x_trans, y_trans, 0]
        return self.trans

    def translate_randomly(self, beam_focus_size):
        """
        Translate randomly.
        :param beam_focus_size: Radius within which we want our translation
        :return: None
        """

        new_pos = get_random_translations(self.atom_pos, beam_focus_size)
        trans_temp = new_pos - self.atom_pos
        self.trans = [trans_temp[10][0], trans_temp[10][1], trans_temp[10][2]]
        self.set_atom_pos(new_pos)

    # setters and getters
    def set_atom_pos(self, pos):
        self.atom_pos = pos

    def get_atom_pos(self):
        return self.atom_pos

    def get_num_atoms(self):
        return self.atom_pos.shape[0]

    def get_num_compton_q_samples(self):
        return self.num_compton_q_samples
    
    def get_q_sample(self):
        return self.q_sample
        
    def read_h5file(self, fname, datasetname):
        """
        Parse the h5file to get the particle position and the other information

        :param fname: The file name of the h5file
        :param datasetname: The dataset name to parse
        :return:
        """
        with h5py.File(fname, 'r') as f:
            atom_pos = f.get(datasetname + '/r').value  # atom position -> N x 3 array
            ion_list = f.get(
                datasetname + '/xyz').value  # length = N, contain atom type id for each atom
            self.atom_pos = atom_pos[np.argsort(ion_list)]
            _, idx = np.unique(np.sort(ion_list), return_index=True)
            self.split_idx = np.append(idx, [len(ion_list)])

            # get atom factor table, sorted by atom type id
            atom_type = f.get(
                datasetname + '/T').value  # atom type array, each type is represented by an integer
            self.num_atom_types = len(atom_type)
            ff_table = f.get(datasetname + '/ff').value
            self.ff_table = ff_table[np.argsort(atom_type)]

            self.q_sample = f.get(datasetname + '/halfQ').value
            self.num_q_samples = len(self.q_sample)
            self.compton_q_sample = f.get(datasetname + '/Sq_halfQ').value
            self.num_compton_q_samples = len(self.compton_q_sample)
            self.sBound = f.get(datasetname + '/Sq_bound').value
            self.nFree = f.get(datasetname + '/Sq_free').value

    def read_pdb(self, fname, ff='WK'):
        """
        Get particle information from reading pdb file.
        Implement the necessary transformation to different chains of the particle based on
        the symmetry specified in the pdb file(REMARK 350 BIOMT).
        Set the ff_table and q_sample manually.

        :param fname: The file name of the pdb file to read
        :param ff: The form factor table to use
        :return:
        """

        atoms,atomslist = symmpdb(fname)
        
        
        #print atomslist
       
        
        xpos = [row[0] for row in atomslist]
        ypos = [row[1] for row  in atomslist]
        zpos = [row[2] for row in atomslist]
        an =[row[3] for row in atomslist]
        self.atom_struct = np.array([xpos,ypos,zpos,an])
        self.atomic_symbol = [row[4] for row in atomslist]
        self.atomic_variant = [row[5] for row in atomslist]
        self.residue = [row[6] for row in atomslist]
        self.atom_pos = atoms[:, 0:3] / 10 ** 10  # convert unit from Angstroms to m
        self.at = atoms[:,3]
        tmp = (100 * atoms[:, 3] + atoms[:, 4]).astype(
            int)  # hack to get split idx from the sorted atom array
        atom_type, idx = np.unique(np.sort(tmp), return_index=True)
        self.num_atom_types = len(atom_type)
        self.split_idx = np.append(idx, [len(tmp)])

        bohr_radius = 0.529177206


        if ff == 'WK':
            """
            Here, one tries to calculate the form factor from formula and tables.
            Therefore, one needs to setup some reference points for interpolation.
            Here, the qs variable is such a variable containing the momentum length
            at which one calculate the reference values.
            """
            # set up q samples and compton
            qs = np.linspace(0, 10, 101) / (2.0 * np.pi * bohr_radius * 2.0)
            self.q_sample = qs
            self.compton_q_sample = qs
            self.num_q_samples = len(qs)
            self.num_compton_q_samples = len(qs)
            self.sBound = np.zeros(self.num_q_samples)
            self.nFree = np.zeros(self.num_q_samples)

            # calculate form factor using WaasKirf coeffs table
            wk_dbase = load_waaskirf_database()
            for i in idx:
                if i == 0:
                    zz = int(atoms[i, 3])  # atom type
                    qq = int(atoms[i, 4])  # charge
                    idx1 = np.where(wk_dbase[:, 0] == zz)[0]
                    flag = True
                    for j in idx1:
                        if int(wk_dbase[j, 1]) == qq:
                            [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5] = wk_dbase[j, 2:]
                            self.ff_table = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                                             a2 * np.exp(-b2 * self.q_sample ** 2) +
                                             a3 * np.exp(-b3 * self.q_sample ** 2) +
                                             a4 * np.exp(-b4 * self.q_sample ** 2) +
                                             a5 * np.exp(-b5 * self.q_sample ** 2) + c)
                            flag = False
                            break
                    if flag:
                        print('Atom number = ' + str(zz) + ' with charge ' + str(qq))
                        raise ValueError('Unrecognized atom type!')
                else:
                    zz = int(atoms[i, 3])  # atom type
                    qq = int(atoms[i, 4])  # charge
                    idx1 = np.where(wk_dbase[:, 0] == zz)[0]
                    flag = True
                    for j in idx1:
                        if int(wk_dbase[j, 1]) == qq:
                            [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5] = wk_dbase[j, 2:]

                            ff = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                                  a2 * np.exp(-b2 * self.q_sample ** 2) +
                                  a3 * np.exp(-b3 * self.q_sample ** 2) +
                                  a4 * np.exp(-b4 * self.q_sample ** 2) +
                                  a5 * np.exp(-b5 * self.q_sample ** 2) + c)

                            self.ff_table = np.vstack((self.ff_table, ff))
                            flag = False
                            break
                    if flag:
                        print('Atom number = ' + str(zz) + ' with charge ' + str(qq))
                        raise ValueError('Unrecognized atom type!')

        elif ff == 'pmi':
            # set up ff table
            ffdbase = load_ff_database()
            for i in idx:
                if i == 0:
                    zz = int(atoms[i, 3])  # atom type
                    qq = int(atoms[i, 4])  # charge
                    self.ff_table = ffdbase[:, zz] * (zz - qq) / (zz * 1.0)
                else:
                    zz = int(atoms[i, 3])  # atom type
                    qq = int(atoms[i, 4])  # charge
                    self.ff_table = np.vstack(
                        (self.ff_table, ffdbase[:, zz] * (zz - qq) / (zz * 1.0)))

            # set up q samples and compton
            self.q_sample = ffdbase[:, 0] / (2.0 * np.pi * 0.529177206 * 2.0)
            self.compton_q_sample = ffdbase[:, 0] / (2.0 * np.pi * 0.529177206 * 2.0)
            self.num_q_samples = len(ffdbase[:, 0])
            self.num_compton_q_samples = len(ffdbase[:, 0])
            self.sBound = np.zeros(self.num_q_samples)
            self.nFree = np.zeros(self.num_q_samples)
        elif ff == 'CM':
            """
            Here, one tries to calculate the form factor from formula and tables.
            Therefore, one needs to setup some reference points for interpolation.
            Here, the qs variable is such a variable containing the momentum length
            at which one calculate the reference values.
            """
            # set up q samples and compton
            qs = np.linspace(0, 10, 101) / (2.0 * np.pi * bohr_radius * 2.0)
            self.q_sample = qs
            self.compton_q_sample = qs
            self.num_q_samples = len(qs)
            self.num_compton_q_samples = len(qs)
            self.sBound = np.zeros(self.num_q_samples)
            self.nFree = np.zeros(self.num_q_samples)
            self.ff_table = None
            
            # calculate form factor using WaasKirf coeffs table
            cm_dbase = load_cromermann_database()

            for i in idx:
                zz = int(atoms[i, 3])  # atom type
                
                qq = int(atoms[i, 4])  # charge
                idx1 = np.where(cm_dbase[:, 0] == zz)[0]
                flag = True
                for j in idx1:
                    if int(cm_dbase[j, 1]) == qq:
                        [a1, a2, a3, a4, c, b1, b2, b3, b4] = cm_dbase[j, 2:]
                        if self.ff_table is None:
                            self.ff_table = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                                             a2 * np.exp(-b2 * self.q_sample ** 2) +
                                             a3 * np.exp(-b3 * self.q_sample ** 2) +
                                             a4 * np.exp(-b4 * self.q_sample ** 2) + c)
                        else:
                            ff = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                                  a2 * np.exp(-b2 * self.q_sample ** 2) +
                                  a3 * np.exp(-b3 * self.q_sample ** 2) +
                                  a4 * np.exp(-b4 * self.q_sample ** 2) + c)

                            self.ff_table = np.vstack((self.ff_table, ff))
                        flag = False
                        break
                if flag:
                    print('Atom number = ' + str(zz) + ' with charge ' + str(qq))
                    raise ValueError('Unrecognized atom type!')
        else:
            raise ValueError('Unrecognized form factor source!')

    def create_from_atoms(self, atoms):
        atom_types = {'H': 1, 'HE': 2, 'C': 6, 'N1+': 6, 'N': 7, 'O': 8, 'O1-': 9, 'P': 15, 'S': 16, 'CL': 18, 'FE': 26}
        
        all_atoms = []
        for atom_info in atoms:
            for info in atom_info:
                if type(info) == str:
                    atom = info
                elif len(info) == 3:
                    coordinates = info
                else:
                    raise ValueError('Invalid atom information!')
            atomic_number = atom_types[atom]
            total_atom = [coordinates[0], coordinates[1], coordinates[2], atomic_number, 0]
            all_atoms.append(total_atom)                                      # charge = 0 (by default)
        atoms = np.asarray(all_atoms)

        self.atom_pos = atoms[:, 0:3] * 1e-10
        tmp = (100 * atoms[:, 3] + atoms[:, 4]).astype(int)
        atom_type, idx = np.unique(np.sort(tmp), return_index=True)
        
        self.num_atom_types = len(atom_type)
        self.split_idx = np.append(idx, [len(tmp)])
        qs = np.linspace(0, 10, 101) / (2.0 * np.pi * 0.529177206 * 2.0)
        self.q_sample = qs
        self.compton_q_sample = qs
        self.num_q_samples = len(qs)
        self.sBound = np.zeros(self.num_q_samples)
        self.nFree = np.zeros(self.num_q_samples)

        wk_dbase = load_waaskirf_database()
        for i in idx:
            if i == 0:
                zz = atoms[i, 3]
                qq = atoms[i, 4]
                idx1 = np.where(wk_dbase[:, 0] == zz)[0]
                flag = True
                for j in idx1:
                    if int(wk_dbase[j, 1]) == qq:
                        [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5] = wk_dbase[j, 2:]
                        self.ff_table = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                                         a2 * np.exp(-b2 * self.q_sample ** 2) +
                                         a3 * np.exp(-b3 * self.q_sample ** 2) +
                                         a4 * np.exp(-b4 * self.q_sample ** 2) +
                                         a5 * np.exp(-b5 * self.q_sample ** 2) + c)
                        flag = False
                        break
                        if flag:
                            print('Atom number = ' + str(zz) + ' with charge ' + str(qq))
                            raise ValueError('Unrecognized atom type!')
            else:
                zz = int(atoms[i, 3])  # atom type
                qq = int(atoms[i, 4])  # charge
                idx1 = np.where(wk_dbase[:, 0] == zz)[0]
                flag = True
                for j in idx1:
                    if int(wk_dbase[j, 1]) == qq:
                        # print "Enter: ", j
                        [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5] = wk_dbase[j, 2:]

                        ff = (a1 * np.exp(-b1 * self.q_sample ** 2) +
                              a2 * np.exp(-b2 * self.q_sample ** 2) +
                              a3 * np.exp(-b3 * self.q_sample ** 2) +
                              a4 * np.exp(-b4 * self.q_sample ** 2) +
                              a5 * np.exp(-b5 * self.q_sample ** 2) + c)
                        self.ff_table = np.vstack((self.ff_table, ff))
                        flag = False
                        break
                    if flag:
                        print('Atom number = ' + str(zz) + ' with charge ' + str(qq))
                        raise ValueError('Unrecognized atom type!')


def rotate_particle(quaternion, particle):
    """
    Apply one quaternion to rotate the particle.

    :param quaternion:
    :param particle:
    :return:
    """
    rot3d = quaternion2rot3d(quaternion)
    new_pos = np.dot(particle.atom_pos, rot3d.T)
    particle.set_atom_pos(new_pos)
