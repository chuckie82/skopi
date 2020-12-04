import numpy as np
import os
import pytest

import pysingfel as ps
import pysingfel.gpu as pg
import pysingfel.constants as cst
from pysingfel.build_autoranging_frames import BuildAutoRangeFrames

import six
if six.PY2:
    PSCalib = pytest.importorskip("PSCalib")
if six.PY3:
    psana = pytest.importorskip("psana")


class TestAutorangingDetector(object):
    """Test autoranging detector functions."""
    @classmethod
    def setup_class(cls):
        ex_dir_ = os.path.dirname(__file__) + '/../../../examples'

        # Load beam
        beam = ps.Beam(ex_dir_+'/input/beam/amo86615.beam')

        # Load and initialize the detector
        np.random.seed(0)
        det = ps.Epix10kDetector(
            geom=ex_dir_+'/input/lcls/xcsx35617/'
                 'Epix10ka2M::CalibV1/XcsEndstation.0:Epix10ka2M.0/geometry/0-end.data',
            run_num=0,
            beam=beam,
            cameraConfig='fixedMedium')
        cls.det = det
            
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

        # Flat Field
        cls.flatField = np.ones((cls.det.panel_num, cls.det.panel_pixel_num_x[0], cls.det.panel_pixel_num_y[0]))*1.0
        cls.I0width = 0.03
        cls.I0min = 0
        cls.I0max = 150000
        cls.bauf = BuildAutoRangeFrames(cls.det, cls.I0width, cls.I0min, cls.I0max, cls.flatField)
        cls.bauf.makeFrame()    

    def test_add_phase_shift(self):
        """Test phase shift from translation."""
        pattern = self.det.add_phase_shift(self.pattern_0, self.part_coord_1)
        assert np.allclose(pattern, self.pattern_1)

    @pytest.mark.skipif(six.PY3, reason="Calibration constants require psana2")
    def test_pedestal_nonzero(self):
        """Test existence of pedestals."""
        assert np.sum(abs(self.det.pedestals[:])) > np.finfo(float).eps

    def test_flatfield(self):
        """Test a certain pixel value given a flat field."""
        assert self.bauf.frame[10][0][383] == 1.0014666654326363
