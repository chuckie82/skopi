import numpy as np
import os

import pysingfel as ps
import pysingfel.gpu as pg
import pysingfel.constants as cst


class TestDiffractionPattern(object):
    """Test calculate_diffraction_pattern_gpu."""
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../examples'

        # Load beam
        beam = ps.Beam(ex_dir_+'/input/exp_chuck.beam')

        # Load and initialize the detector
        det = ps.PnccdDetector(
            geom=ex_dir_+'/lcls/amo86615/'
                 'PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
            beam=beam)

        cls.mesh_length = 15
        cls.mesh, voxel_length = det.get_reciprocal_mesh(
            voxel_number_1d=cls.mesh_length)

        # 1 Atom
        cls.particle_1 = ps.Particle()
        cls.particle_1.create_from_atoms([
            ("O", np.array([0., 0., 0.]))
        ])
        cls.volume_1 = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_1)

        # 2 Atoms x
        cls.particle_2x = ps.Particle()
        cls.particle_2x.create_from_atoms([
            ("O", cst.vecx),
            ("O", -cst.vecx)
        ])
        cls.volume_2x = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2x)

        # 2 Atoms y
        cls.particle_2y = ps.Particle()
        cls.particle_2y.create_from_atoms([
            ("O", cst.vecy),
            ("O", -cst.vecy)
        ])
        cls.volume_2y = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2y)

        # 2 Atoms z
        cls.particle_2z = ps.Particle()
        cls.particle_2z.create_from_atoms([
            ("O", cst.vecz),
            ("O", -cst.vecz)
        ])
        cls.volume_2z = pg.calculate_diffraction_pattern_gpu(
            cls.mesh, cls.particle_2z)

    def test_pattern_1_h_symetry(self):
        """Test h symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, self.volume_1[::-1, :, :])

    def test_pattern_1_k_symetry(self):
        """Test k symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, self.volume_1[:, ::-1, :])

    def test_pattern_1_l_symetry(self):
        """Test l symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, self.volume_1[:, :, ::-1])

    def test_pattern_1_hk_symetry(self):
        """Test h-k symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, np.swapaxes(self.volume_1, 0, 1))

    def test_pattern_1_kl_symetry(self):
        """Test k-l symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, np.swapaxes(self.volume_1, 1, 2))

    def test_pattern_1_lh_symetry(self):
        """Test l-h symmetry of volume with 1 atom."""
        assert np.allclose(self.volume_1, np.swapaxes(self.volume_1, 2, 0))

    def test_pattern_2x_h_symetry(self):
        """Test h symmetry of volume with 2 atoms along x."""
        assert np.allclose(self.volume_2x, self.volume_2x[::-1, :, :])

    def test_pattern_2x_k_symetry(self):
        """Test k symmetry of volume with 2 atoms along x."""
        assert np.allclose(self.volume_2x, self.volume_2x[:, ::-1, :])

    def test_pattern_2x_l_symetry(self):
        """Test l symmetry of volume with 2 atoms along x."""
        assert np.allclose(self.volume_2x, self.volume_2x[:, :, ::-1])

    def test_pattern_2z_hk_symetry(self):
        """Test h-k symmetry of volume with 2 atoms along z."""
        assert np.allclose(self.volume_2z, np.swapaxes(self.volume_2z, 0, 1))

    def test_pattern_2x_kl_symetry(self):
        """Test k-l symmetry of volume with 2 atoms along x."""
        assert np.allclose(self.volume_2x, np.swapaxes(self.volume_2x, 1, 2))

    def test_pattern_2y_lh_symetry(self):
        """Test l-h symmetry of volume with 2 atoms along y."""
        assert np.allclose(self.volume_2y, np.swapaxes(self.volume_2y, 2, 0))

    def test_pattern_1_vs_2x_particles(self):
        """Test diffraction equivalence of 1-2 atoms along x axis."""
        assert np.allclose(
            (self.volume_2x - 4*self.volume_1)[self.mesh_length//2, :, :], 0.)

    def test_pattern_1_vs_2y_particles(self):
        """Test diffraction equivalence of 1-2 atoms along y axis."""
        assert np.allclose(
            (self.volume_2y - 4*self.volume_1)[:, self.mesh_length//2, :], 0.)

    def test_pattern_1_vs_2z_particles(self):
        """Test diffraction equivalence of 1-2 atoms along z axis."""
        assert np.allclose(
            (self.volume_2z - 4*self.volume_1)[:, :, self.mesh_length//2], 0.)
