import numpy as np
import os
import sys

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc
from pysingfel.util import xp, asnumpy

from .base import DetectorBase


class SimpleSquareDetector(DetectorBase):
    """
    Class for simple square detector.
    """

    def __init__(self, N_pixel, det_size, det_distance, beam=None):
        """
        Initialize the detector
        :param geom:  The dictionary containing all the necessary information to initialize
                        the detector.
        :param beam: The beam object
        """
        super(SimpleSquareDetector, self).__init__()

        ar = xp.arange(N_pixel)
        sep = float(det_size) / N_pixel
        x = -det_size/2 + sep/2 + ar*sep
        y = -det_size/2 + sep/2 + ar*sep
        X, Y = xp.meshgrid(x, y, indexing='xy')
        Xar, Yar = xp.meshgrid(ar, ar, indexing='xy')

        self.panel_num = 1
        self.panel_pixel_num_x = N_pixel
        self.panel_pixel_num_y = N_pixel
        self._shape = (1, N_pixel, N_pixel)

        # Define all properties the detector should have
        self._distance = det_distance
        self.center_z = self._distance * xp.ones(self._shape, dtype=xp.float64)

        p_center_x = xp.stack((X,))
        p_center_y = xp.stack((Y,))
        self.pixel_width = sep * xp.ones(self._shape, dtype=xp.float64)
        self.pixel_height = sep * xp.ones(self._shape, dtype=xp.float64)
        self.center_x = p_center_x
        self.center_y = p_center_y

        # construct the the pixel position array
        self.pixel_position = xp.zeros(self._shape + (3,))
        self.pixel_position[:, :, :, 0] = self.center_x
        self.pixel_position[:, :, :, 1] = self.center_y
        self.pixel_position[:, :, :, 2] = self.center_z

        # Pixel map
        p_map_x = xp.stack((Xar,))
        p_map_y = xp.stack((Yar,))
        # [panel number, pixel num x, pixel num y]
        self.pixel_index_map = xp.stack((p_map_x, p_map_y), axis=-1)
        # Detector pixel number info
        self.detector_pixel_num_x = asnumpy(xp.max(self.pixel_index_map[:, :, :, 0]))
        self.detector_pixel_num_y = asnumpy(xp.max(self.pixel_index_map[:, :, :, 1]))

        # Panel pixel number info
        # number of pixels in each panel in x/y direction
        self.panel_pixel_num_x = self.pixel_index_map.shape[1]
        self.panel_pixel_num_y = self.pixel_index_map.shape[2]

        # total number of pixels (px*py)
        self.pixel_num_total = np.prod(self._shape)

        # Calculate the pixel area
        self.pixel_area = xp.multiply(self.pixel_height, self.pixel_width)

        # Get reciprocal space configurations and corrections.
        self.initialize_pixels_with_beam(beam=beam)
