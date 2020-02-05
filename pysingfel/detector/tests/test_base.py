import numpy as np
import os

import pysingfel as ps
import pysingfel.gpu as pg
import pysingfel.constants as cst


class TestAddPhaseShift(object):
    """Test add_phase_shift."""
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../../examples'

        # Load beam
        beam = ps.Beam(ex_dir_+'/input/exp_chuck.beam')

        # Load and initialize the detector
        det = ps.PnccdDetector(
            geom=ex_dir_+'/lcls/amo86615/'
                 'PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
            beam=beam)
        cls.det = det

        cls.pos_recip = det.pixel_position_reciprocal

        # Ref Particle
        cls.particle_0 = ps.Particle()
        cls.particle_0.create_from_atoms([
            ("O", cst.vecx),
            ("O", 2*cst.vecy),
            ("O", 3*cst.vecz),
        ])
        cls.pattern_0 = pg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_0, return_type="complex_field")

        # Second Particle
        cls.part_coord_1 = np.array((0.5, 0.2, 0.1))
        cls.particle_1 = ps.Particle()
        cls.particle_1.create_from_atoms([
            ("O", cst.vecx + cls.part_coord_1),
            ("O", 2*cst.vecy + cls.part_coord_1),
            ("O", 3*cst.vecz + cls.part_coord_1),
        ])
        cls.pattern_1 = pg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_1, return_type="complex_field")

    def test_add_phase_shift(self):
        """Test phase shift from translation."""
        pattern = self.det.add_phase_shift(self.pattern_0, self.part_coord_1)
        assert np.allclose(pattern, self.pattern_1)
