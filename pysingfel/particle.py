import h5py
import numpy as np
import itertools as itertools
from scipy import ndimage
from matplotlib import pyplot as plt
from pysingfel.util import symmpdb
from pysingfel.geometry import quaternion2rot3d, get_random_rotation, get_random_translations
from pysingfel.ff_waaskirf_database import *


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
        
        # Masks and solvent
        self.solvent_mean_electron_density = 0.334 * 10**30 # in e/m**3
        self.hydration_layer_thickness = 4.0 / 10**10    # in meter
        self.mesh_voxel_size           = 2.0 / 10**10    # in meter
        self.mesh = None         # real space mesh for mask definitions
        self.solvent_mask = None 
        self.solute_mask = None 

        # Normal Mode Analysis
        self.normal_mode_vectors = None
        self.normal_mode_variances = None
        self.num_normal_modes = 10
        self.elastic_network_cutoff = 6.  # in Angstroem

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
    
    def set_hydration_layer_thickness(self, hydration_layer_thickness):
        self.hydration_layer_thickness = hydration_layer_thickness

    def set_mesh_voxel_size(self, mesh_voxel_size):
        self.mesh_voxel_size = mesh_voxel_size

    def set_num_normal_modes(self, num_normal_modes):
        self.num_normal_modes = num_normal_modes

    def set_elastic_network_cutoff(self, elastic_network_cutoff):
        self.elastic_network_cutoff = elastic_network_cutoff  # in Angstroem
        
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
        atoms,atomslist = symmpdb(fname, ff)
        xpos = [row[0] for row in atomslist]
        ypos = [row[1] for row  in atomslist]
        zpos = [row[2] for row in atomslist]
        an =[row[3] for row in atomslist] # atomic number
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
        bohr_radius = 0.529177206
        qs = np.linspace(0, 10, 101) / (2.0 * np.pi * bohr_radius  * 2.0)
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
    
    #### MASKS AND MESHES ####
    
    def create_masks(self):
        """create_masks
        """
        self.mesh         = self.build_particle_mesh()
        self.solute_mask  = self.create_solute_mask(dry=True)
        self.solvent_mask = self.solute_mask * ~self.create_solute_mask(dry=False)

    def show_masks(self):
        if self.mesh is None:
            print('... masks not created yet ...')
        else:
            islice = np.floor(self.mesh.shape[0]/2).astype('int')
            
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6,9), sharex=True,  sharey=True, dpi=180)

            axes[0,0].set_title('Solute mask')
            axes[0,0].set_ylabel('YZ central slice')
            axes[0,0].imshow(self.solute_mask[islice,...]*1, cmap='Greys_r')
            axes[1,0].set_ylabel('XZ central slice')
            axes[1,0].imshow(self.solute_mask[:,islice,:]*1, cmap='Greys_r')
            axes[2,0].set_xlabel('voxel index')
            axes[2,0].set_ylabel('XY central slice')
            axes[2,0].imshow(self.solute_mask[:,:,islice]*1, cmap='Greys_r')

            axes[0,1].set_title('Solvent mask')
            axes[0,1].imshow(self.solvent_mask[islice,...]*1, cmap='Blues')
            axes[1,1].imshow(self.solvent_mask[:,islice,:]*1, cmap='Blues')
            axes[2,1].set_xlabel('voxel index')
            axes[2,1].imshow(self.solvent_mask[:,:,islice]*1, cmap='Blues')

            plt.tight_layout()
            plt.show()

    def build_particle_mesh(self):
        """build_particle_mesh:
        Cubic mesh of length set by maximal solute dimension + hydration layers.
        Even number of voxels per side (hence odd number of vertices).
                .---.---.---.---.
                |   |   |   |   |  o vertex that can be used to index vortex
                | x | x | x | x |  . vertex that can not be used as a vortex index
                |/  |/  |/  |/  |  x voxel center indexed by previous (/) vertex
                o---o---o---o---.
                |   |   |   |   |  NOTE: we save the positions of x not o
                | x | x | x | x |
                |/  |/  |/  |/  |
        center> o---o---o---o---.
                |   |   |   |   |
                | x | x | x | x |
                |/  |/  |/  |/  |
                o---o---o---o---.
                |   |   |   |   |
                | x | x | x | x |
                |/  |/  |/  |/  |
                o---o---o---o---.
                        ^
                        |
                     center
        """

        particle_dimension = np.zeros(3)
        for i in range(3):
            particle_dimension[i] = (np.max(self.atom_pos[:,i]) -
                                     np.min(self.atom_pos[:,i]))
        
        mesh_length = (np.max(particle_dimension) +
                       4*self.hydration_layer_thickness)
        mesh_vertex_number_1d = np.ceil(mesh_length / self.mesh_voxel_size)
        mesh_length = (mesh_vertex_number_1d - 1) * self.mesh_voxel_size
        if not mesh_vertex_number_1d % 2:
            mesh_length           += self.mesh_voxel_size
            mesh_vertex_number_1d += 1

        linspace = np.linspace(-mesh_length/2.0, 
                                mesh_length/2.0, 
                                mesh_vertex_number_1d)
        mesh_stack = np.asarray(np.meshgrid(linspace, linspace, linspace, indexing='ij'))
        mesh_stack = np.moveaxis(mesh_stack, 0, -1)

        center = self.get_particle_center()
        for i in range(3):
            mesh_stack[...,i] += center[i]
            mesh_stack[...,i] += self.mesh_voxel_size / 2.

        return mesh_stack

    def get_particle_center(self):
        """get_particle_center
        """
        center = np.zeros(3)
        for i in np.arange(3):
            center[i] = 0.5*(np.max(self.atom_pos[:,i]) +
                             np.min(self.atom_pos[:,i]))
        return center

    def create_solute_mask(self, dry=True):
        """create_solute_mask
        """
        mask      = self.initialize_solute_mask()
        mask      = self.dilate_solute_mask(mask, dry=dry)
        return mask

    def initialize_solute_mask(self):
        """initialize_solute_mask
        """
        mask = np.ones(self.mesh.shape[:3], dtype='bool')

        atom_type_num    = len(self.split_idx) - 1
        split_index      = np.array(self.split_idx)
        atom_voxel_index = np.zeros(3, dtype='int')
        for atom_type in range(atom_type_num):
            for atom_iter in range(split_index[atom_type], split_index[atom_type + 1]):
                for i in range(3):
                    atom_voxel_index[i] = np.floor(
                                            (self.atom_pos[atom_iter,i] - self.mesh[0,0,0,i]) / self.mesh_voxel_size
                                          )
                mask[atom_voxel_index[0],
                     atom_voxel_index[1],
                     atom_voxel_index[2]] = False
        return mask

    def dilate_solute_mask(self, mask, dry=True):
        """dilate_solute_mask
        """
        sphere_radius = 3.5 / 10**10 # in meter
        if not dry:
            sphere_radius += self.hydration_layer_thickness
        sphere = self.build_mask_sphere(sphere_radius)
        mask = ~ndimage.binary_closing(~mask, structure=sphere)
        mask = ~ndimage.binary_dilation(~mask, structure=sphere)
        return mask

    def build_mask_sphere(self, sphere_radius):
        """build_mask_sphere
        """
        sphere_vertex_number_1d = np.ceil(2.0 * sphere_radius / self.mesh_voxel_size).astype('int')
        sphere_element = ndimage.generate_binary_structure(3,1)
        sphere = ndimage.iterate_structure(sphere_element, np.ceil(sphere_vertex_number_1d / 3).astype('int'))
        return sphere

    #### DYNAMICS ####

    def gen_normal_modes(self):
        """gen_normal_modes
        """
        print('>>> Computing normal modes with ProDy')
        from prody import ANM as prody_ANM
        from prody import confProDy

        confProDy(verbosity='critical')

        anm = prody_ANM()
        anm.buildHessian(self.atom_pos * 10**10, cutoff=self.elastic_network_cutoff)
        anm.calcModes(n_modes=self.num_normal_modes)

        self.normal_mode_vectors = anm.getEigvecs().reshape(self.atom_pos.shape[0],
                                                            self.atom_pos.shape[1],
                                                            self.num_normal_modes)
        self.normal_mode_variances = 1./anm.getEigvals()

    def update_conformation(self, rmsd=3.):
        """update_conformation
        """

        latent_coordinates = rmsd * self.get_random_latent_coordinates()

        deformation_vector = np.zeros(self.atom_pos.shape)
        for i in range(self.num_normal_modes):
            deformation_vector += (latent_coordinates[i] * 
                                   np.sqrt(self.normal_mode_variances[i]) *
                                   self.normal_mode_vectors[...,i])
        deformation_vector /= 10**10 # back to meter

        return self.atom_pos + deformation_vector

    def get_random_latent_coordinates(self):
        """get_random_latent_coordinates
        Outputs a set of latent_coordinates that together would lead to a deformation
        from the initial structure with a RMSD of 1 Angstroem
        """
        
        latent_coordinates = np.random.randn(self.num_normal_modes)
        
        scale_factor = 0.
        for i in range(self.num_normal_modes):
            scale_factor += latent_coordinates[i]**2 * self.normal_mode_variances[i]
        scale_factor = np.sqrt(self.atom_pos.shape[0]) / np.sqrt(scale_factor)
        
        return scale_factor * latent_coordinates

