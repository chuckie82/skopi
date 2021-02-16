import numpy as np
import os
import pytest

import skopi as sk
import skopi.gpu as sg
import skopi.constants as cst
from skopi.util import xp

import six
if six.PY2:
    PSCalib = pytest.importorskip("PSCalib")
if six.PY3:
    psana = pytest.importorskip("psana")


class TestDetectorBase(object):
    """Test base detector functions."""
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../../examples'

        # Load beam
        beam = sk.Beam(ex_dir_+'/input/beam/amo86615.beam')

        # Load and initialize the detector
        det = sk.PnccdDetector(
            geom=ex_dir_+'/input/lcls/amo86615/'
                 'PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
            beam=beam)
        cls.det = det

        cls.pos_recip = det.pixel_position_reciprocal

        # Ref Particle
        cls.particle_0 = sk.Particle()
        cls.particle_0.create_from_atoms([  # Angstrom
            ("O", cst.vecx),
            ("O", 2*cst.vecy),
            ("O", 3*cst.vecz),
        ])
        cls.pattern_0 = sg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_0, return_type="complex_field")

        # Second Particle
        cls.part_coord_1 = np.array((0.5, 0.2, 0.1))  # Angstrom
        cls.particle_1 = sk.Particle()
        cls.particle_1.create_from_atoms([  # Angstrom
            ("O", cst.vecx + cls.part_coord_1),
            ("O", 2*cst.vecy + cls.part_coord_1),
            ("O", 3*cst.vecz + cls.part_coord_1),
        ])
        cls.part_coord_1 *= 1e-10  # Angstrom -> meter
        cls.pattern_1 = sg.calculate_diffraction_pattern_gpu(
            cls.pos_recip, cls.particle_1, return_type="complex_field")

    def test_add_phase_shift(self):
        """Test phase shift from translation."""
        pattern = self.det.add_phase_shift(self.pattern_0, self.part_coord_1)
        assert np.allclose(pattern, self.pattern_1)

    def test_pedestal_nonzero(self):
        """Test existence of pedestals."""
        assert np.sum(abs(self.det.pedestals[:])) > np.finfo(float).eps


def test_distance_change():
    """Test distance property change."""
    det = sk.SimpleSquareDetector(
        N_pixel=1024, det_size=0.1, det_distance=0.2)
    distance_1 = det.distance
    pixel_position_1 = det.pixel_position.copy()
    det.distance *= 2
    assert np.isclose(det.distance, distance_1*2)
    assert xp.allclose(det.pixel_position[..., 0],
                       pixel_position_1[..., 0])  # X unchanged
    assert xp.allclose(det.pixel_position[..., 1],
                       pixel_position_1[..., 1])  # Y unchanged
    assert xp.allclose(det.pixel_position[..., 2],
                       pixel_position_1[..., 2]*2)  # Z doubled
