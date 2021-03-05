import numpy as np
import os
import pytest

import skopi as sk
import skopi.gpu as pg
import skopi.constants as cst
from skopi.util import xp

import six
if six.PY2:
    PSCalib = pytest.importorskip("PSCalib")
if six.PY3:
    psana = pytest.importorskip("psana")

class TestDiffractionPattern(object):
    """Test calculate_diffraction_pattern_gpu based on expected symmetry."""
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../examples'

        # Load beam
        beam = sk.Beam(ex_dir_+'/input/beam/amo86615.beam')

        # Load and initialize the detector
        det = sk.PnccdDetector(
            geom=ex_dir_+'/input/lcls/amo86615/'
                 'PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
            beam=beam)

        cls.mesh_length = 15
        cls.mesh, voxel_length = det.get_reciprocal_mesh(
            voxel_number_1d=cls.mesh_length)

        # 1 Atom
        cls.particle_1 = sk.Particle()
        cls.particle_1.create_from_atoms([
            ("O", np.array([0., 0., 0.]))
        ])
        cls.volume_1 = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_1)

        # 2 Atoms x
        cls.particle_2x = sk.Particle()
        cls.particle_2x.create_from_atoms([
            ("O", cst.vecx),
            ("O", -cst.vecx)
        ])
        cls.volume_2x = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2x)

        # 2 Atoms y
        cls.particle_2y = sk.Particle()
        cls.particle_2y.create_from_atoms([
            ("O", cst.vecy),
            ("O", -cst.vecy)
        ])
        cls.volume_2y = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2y)

        # 2 Atoms z
        cls.particle_2z = sk.Particle()
        cls.particle_2z.create_from_atoms([
            ("O", cst.vecz),
            ("O", -cst.vecz)
        ])
        cls.volume_2z = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2z)

    def test_pattern_1_h_symetry(self):
        """Test h symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, self.volume_1[::-1, :, :])

    def test_pattern_1_k_symetry(self):
        """Test k symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, self.volume_1[:, ::-1, :])

    def test_pattern_1_l_symetry(self):
        """Test l symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, self.volume_1[:, :, ::-1])

    def test_pattern_1_hk_symetry(self):
        """Test h-k symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, xp.swapaxes(self.volume_1, 0, 1))

    def test_pattern_1_kl_symetry(self):
        """Test k-l symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, xp.swapaxes(self.volume_1, 1, 2))

    def test_pattern_1_lh_symetry(self):
        """Test l-h symmetry of volume with 1 atom."""
        assert xp.allclose(self.volume_1, xp.swapaxes(self.volume_1, 2, 0))

    def test_pattern_2x_h_symetry(self):
        """Test h symmetry of volume with 2 atoms along x."""
        assert xp.allclose(self.volume_2x, self.volume_2x[::-1, :, :])

    def test_pattern_2x_k_symetry(self):
        """Test k symmetry of volume with 2 atoms along x."""
        assert xp.allclose(self.volume_2x, self.volume_2x[:, ::-1, :])

    def test_pattern_2x_l_symetry(self):
        """Test l symmetry of volume with 2 atoms along x."""
        assert xp.allclose(self.volume_2x, self.volume_2x[:, :, ::-1])

    def test_pattern_2z_hk_symetry(self):
        """Test h-k symmetry of volume with 2 atoms along z."""
        assert xp.allclose(self.volume_2z, xp.swapaxes(self.volume_2z, 0, 1))

    def test_pattern_2x_kl_symetry(self):
        """Test k-l symmetry of volume with 2 atoms along x."""
        assert xp.allclose(self.volume_2x, xp.swapaxes(self.volume_2x, 1, 2))

    def test_pattern_2y_lh_symetry(self):
        """Test l-h symmetry of volume with 2 atoms along y."""
        assert xp.allclose(self.volume_2y, xp.swapaxes(self.volume_2y, 2, 0))

    def test_pattern_1_vs_2x_particles(self):
        """Test diffraction equivalence of 1-2 atoms along x axis."""
        assert xp.allclose(
            (self.volume_2x - 4*self.volume_1)[self.mesh_length//2, :, :], 0.)

    def test_pattern_1_vs_2y_particles(self):
        """Test diffraction equivalence of 1-2 atoms along y axis."""
        assert xp.allclose(
            (self.volume_2y - 4*self.volume_1)[:, self.mesh_length//2, :], 0.)

    def test_pattern_1_vs_2z_particles(self):
        """Test diffraction equivalence of 1-2 atoms along z axis."""
        assert xp.allclose(
            (self.volume_2z - 4*self.volume_1)[:, :, self.mesh_length//2], 0.)


def reference_diffraction_calculation(qgrid, atomic_positions, fi, fi_indices):
    """
    Reference implementation for computing diffraction.
    
    :param qgrid: array of shape [nvectors, 3] in per Angstrom
    :param atomic_positions: coordinates array of shape [n_atoms, 3] in Angstrom
    :param fi: atomic form factors of shape [n_atom types, nvectors]
    :param fi_indices: atom type indexing array of shape [n_atoms]
    :return pattern: simulated intensities 
    """
    
    I = np.zeros(qgrid.shape[0], dtype=np.complex128)
    for i,qvector in enumerate(qgrid):
        for j in range(atomic_positions.shape[0]):
            ff = fi[fi_indices[j]][i] 
            r = atomic_positions[j,:]
            I[i] += ff * np.sin(np.dot(qvector, r))
            I[i] += 1j * ff * np.cos(np.dot(qvector, r)) 
            
    return np.square(np.abs(I))


class TestDiffractionVolume(object):
    """Test calculate_diffraction_pattern_gpu based on reference calculation for full volume."""
    @classmethod
    def setup_class(self):
        # set up q-grid in Angstroms to test
        q_xx,q_yy,q_zz = np.meshgrid(np.arange(-5,5.2,0.2), np.arange(-5,5.2,0.2), np.arange(-5,5.2,0.2))
        self.qgrid = np.array([q_xx.flatten(), q_yy.flatten(), q_zz.flatten()]).T 
        self.stol = np.linalg.norm(self.qgrid, axis=-1) / (4*np.pi) 

    def generate_pentagon(self,atoms=['C','C','C','S','S']):
        """
        Generate a pentagon in the XYZ plane
    
        :param atoms: list of atom types
        :return particle: pentagon Particle object
        :return xyz: list of atomic positions in Angstrom
        """
        xyz = np.array([[4.000,   0.000,   0.000],
                        [1.236,   3.804,   0.000],
                        [-3.236,   2.351,   0.000],
                        [-3.236,  -2.351,   0.000],
                        [ 1.236,  -3.804,   0.000]])
        p_input = [(a,pos) for a,pos in zip(atoms, xyz)]
    
        particle = sk.Particle()
        particle.create_from_atoms(p_input)
        return particle, xyz

    def test_volume_calculation(self):
        """
        Validation for calculate_diffraction_pattern_gpu using a reference calculation.
        """
    
        fi_indices_list = [np.zeros(5).astype(int), np.array([0,0,0,1,1])]
        atoms_list = [['C','C','C','C','C'], ['C','C','C','S','S']]
    
        for fi_indices,atoms in zip(fi_indices_list, atoms_list):

            # generate a simple 5-atom particle
            particle, xyz = self.generate_pentagon(atoms)

            # compute atomic form factors
            fi = sk.diffraction.calculate_atomic_factor(particle, self.stol, self.stol.shape[0]) 

            # compute reference volume
            I_ref = reference_diffraction_calculation(self.qgrid, xyz, fi, fi_indices)

            # compute skopi volume, which expects s-vectors (q=2*pi*s) in m-1
            reciprocal_space = self.qgrid / (2*np.pi) * 1e10
            I_skopi = pg.calculate_diffraction_pattern_gpu(reciprocal_space, particle, return_type='intensity')

            assert np.allclose(I_ref, I_skopi)
    
        return
