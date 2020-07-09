import numpy as np
import os
import pytest

import pysingfel as ps
import pysingfel.gpu as pg
import pysingfel.constants as cst
from pysingfel.util import xp

import six
if six.PY2:
    PSCalib = pytest.importorskip("PSCalib")
if six.PY3:
    psana = pytest.importorskip("psana")


class TestAutorangingDetector(object):
<<<<<<< HEAD
    """Test base detector functions."""
=======
    """Test autoranging detector functions."""
>>>>>>> dfc4c4497ab77119664484dd7283d69db8b6ae33
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../../examples'

        # Load beam
        beam = ps.Beam(ex_dir_+'/input/beam/amo86615.beam')

        # Load and initialize the detector
<<<<<<< HEAD
        det = ps.PnccdDetector(
            geom=ex_dir_+'/input/lcls/xcsx35617/'
                    'Epix10ka2M::CalibV1/XcsEndstation.0:Epix10ka2M.0/geometry/0-end.data',
=======
        det = ps.Epix10kDetector(
            geom=ex_dir_+'/input/lcls/xcsx35617/'
                 'Epix10ka2M::CalibV1/XcsEndstation.0:Epix10ka2M.0/geometry/0-end.data',
>>>>>>> dfc4c4497ab77119664484dd7283d69db8b6ae33
            run_num=0,
            beam=beam,
            cameraConfig='fixedMedium')
        cls.det = det
<<<<<<< HEAD

=======
>>>>>>> dfc4c4497ab77119664484dd7283d69db8b6ae33
        cls.pos_recip = det.pixel_position_reciprocal

        # Ref Particle
        cls.particle_0 = ps.Particle()
        cls.particle_0.create_from_atoms([  # Angstrom
            ("O", cst.vecx),
            ("O", 2*cst.vecy),
            ("O", 3*cst.vecz),
        ])
        cls.pattern_0 = pg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_0, return_type="complex_field")

        # Second Particle
        cls.part_coord_1 = np.array((0.5, 0.2, 0.1))  # Angstrom
        cls.particle_1 = ps.Particle()
        cls.particle_1.create_from_atoms([  # Angstrom
            ("O", cst.vecx + cls.part_coord_1),
            ("O", 2*cst.vecy + cls.part_coord_1),
            ("O", 3*cst.vecz + cls.part_coord_1),
        ])
        cls.part_coord_1 *= 1e-10  # Angstrom -> meter
        cls.pattern_1 = pg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_1, return_type="complex_field")

    def test_add_phase_shift(self):
        """Test phase shift from translation."""
        pattern = self.det.add_phase_shift(self.pattern_0, self.part_coord_1)
        assert np.allclose(pattern, self.pattern_1)

    def test_pedestal_nonzero(self):
        """Test existence of pedestals."""
        assert np.sum(abs(self.det.pedestals[:])) > np.finfo(float).eps
