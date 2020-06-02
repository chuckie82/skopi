import numpy as np
import os
import pytest

import pysingfel as ps
import pysingfel.gpu as pg
import pysingfel.constants as cst
from pysingfel.util import xp


class TestUserDefinedDetector(object):
    """Test user defined detector functions."""
    @classmethod
    def setup_class(cls):
        cls.det_shape = (2, 256, 256)
        cls.gap_size = 1e-2  # m
        cls.p_map_xpanel_size =  1e-1  # m
        cls.distance = 0.2  # m
        ar = np.arange(256)
        cls.sep = cls.p_map_xpanel_size / 256
        x = cls.gap_size/2 + cls.sep/2 + ar*cls.sep
        y = -cls.p_map_xpanel_size/2 + cls.sep/2 + ar*cls.sep

        X, Y = np.meshgrid(x, y, indexing='ij')
        Xar, Yar = np.meshgrid(ar, ar, indexing='ij')

        p_center_x = np.stack((X-cls.gap_size-cls.p_map_xpanel_size, X))
        p_center_y = np.stack((Y, Y))

        p_map_x = np.stack((Xar, Xar + 256 + int(cls.gap_size/cls.sep)))
        p_map_y = np.stack((Yar, Yar))

        p_map = np.stack((p_map_x, p_map_y), axis=-1)

        det_dict = {
            'panel number': cls.det_shape[0],
            'panel pixel num x': cls.det_shape[1],
            'panel pixel num y': cls.det_shape[2],
            'detector distance': cls.distance,
            'pixel width': cls.sep * np.ones(cls.det_shape),
            'pixel height': cls.sep * np.ones(cls.det_shape),
            'pixel center x': p_center_x,
            'pixel center y': p_center_y,
            'pixel map': p_map,
        }

        cls.det = ps.UserDefinedDetector(geom=det_dict)

    def test_shape(self):
        assert self.det.shape == self.det_shape

    def test_distance(self):
        assert np.isclose(self.det.distance, self.distance)

    def test_pixel_position_shape(self):
        assert self.det.pixel_position.shape == self.det_shape + (3,)

    def test_pixel_position_z(self):
        assert xp.allclose(self.det.pixel_position[..., 2], self.distance)

    def test_pixel_position_x_sep(self):
        x_diff = self.det.pixel_position[:, 1:, :, 0] \
            - self.det.pixel_position[:, :-1, :, 0]
        assert xp.allclose(x_diff, self.sep)

    def test_pixel_position_y_sep(self):
        y_diff = self.det.pixel_position[:, :, 1:, 1] \
            - self.det.pixel_position[:, :, :-1, 1]
        assert xp.allclose(y_diff, self.sep)
