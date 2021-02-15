import numpy as np
import os
import sys

import skopi.geometry as pg
import skopi.util as pu
import skopi.crosstalk as pc
from skopi.util import xp, asnumpy

from .base import DetectorBase


class UserDefinedDetector(DetectorBase):
    """
    Class for user defined detector. The user has to provide the necessary information
    with a dictionary with proper entries to use this class.
    """

    def __init__(self, geom, beam=None):
        """
        Initialize the detector
        :param geom:  The dictionary containing all the necessary information to initialize
                        the detector.
        :param beam: The beam object
        """
        super(UserDefinedDetector, self).__init__()
        self.initialize(geom=geom, beam=beam)

    def initialize(self, geom, beam):
        """
        Initialize the detector with user defined parameters
        :param geom: The dictionary containing all the necessary information to initialized the
                    detector.
        :param beam: The beam object
        :return: None
        """

        """
        Doc:
            To use this class, the user has to provide the necessary information to initialize
             the detector.
            All the necessary entries are listed in the example notebook.
        """

        # 'detector distance': detector distance in (m)

        ##########################################################################################
        # Extract necessary information
        ##########################################################################################

        # Define the hierarchy system. For simplicity, we only use two-layer structure.
        for key in {'panel number', 'panel pixel num x', 'panel pixel num y'}:
            if key not in geom:
                raise KeyError("Missing required '{}' key.".format(key))
        self.panel_num = int(geom['panel number'])
        self.panel_pixel_num_x = int(geom['panel pixel num x'])
        self.panel_pixel_num_y = int(geom['panel pixel num y'])
        self._shape = (self.panel_num, self.panel_pixel_num_x,
                       self.panel_pixel_num_y)

        # Define all properties the detector should have
        self._distance = None
        if 'pixel center z' in geom:
            if 'detector distance' in geom:
                raise ValueError("Please provide one of "
                                 "'pixel center z' or 'detector distance'.")
            self.center_z = xp.asarray(geom['pixel center z'], dtype=xp.float64)
            self._distance = float(self.center_z.mean())
        else:
            if 'detector distance' not in geom:
                KeyError("Missing required 'detector distance' key.")
            self._distance = float(geom['detector distance'])
            self.center_z = self._distance * xp.ones(self._shape,
                                                    dtype=xp.float64)

        # Below: [panel number, pixel num x, pixel num y]  in (m)
        # Change dtype and make numpy/cupy array
        self.pixel_width = xp.asarray(geom['pixel width'], dtype=xp.float64)
        self.pixel_height = xp.asarray(geom['pixel height'], dtype=xp.float64)
        self.center_x = xp.asarray(geom['pixel center x'], dtype=xp.float64)
        self.center_y = xp.asarray(geom['pixel center y'], dtype=xp.float64)
        self.orientation = np.array([0, 0, 1])

        # construct the the pixel position array
        self.pixel_position = xp.zeros(self._shape + (3,))
        self.pixel_position[..., 0] = self.center_x
        self.pixel_position[..., 1] = self.center_y
        self.pixel_position[..., 2] = self.center_z

        # Pixel map
        if 'pixel map' in geom:
            # [panel number, pixel num x, pixel num y]
            self.pixel_index_map = xp.asarray(geom['pixel map'], dtype=xp.int64)
            # Detector pixel number info
            self.detector_pixel_num_x = asnumpy(
                xp.max(self.pixel_index_map[..., 0]) + 1)
            self.detector_pixel_num_y = asnumpy(
                xp.max(self.pixel_index_map[..., 1]) + 1)

            # Panel pixel number info
            # number of pixels in each panel in x/y direction
            self.panel_pixel_num_x = self.pixel_index_map.shape[1]
            self.panel_pixel_num_y = self.pixel_index_map.shape[2]

        # total number of pixels (px*py)
        self.pixel_num_total = np.prod(self._shape)

        ###########################################################################################
        # Do necessary calculation to finishes the initialization
        ###########################################################################################
        # self.geometry currently only work for the pre-defined detectors
        self.geometry = geom

        # Calculate the pixel area
        self.pixel_area = xp.multiply(self.pixel_height, self.pixel_width)

        # Get reciprocal space configurations and corrections.
        self.initialize_pixels_with_beam(beam=beam)

        ##########################################################################################
        # Do necessary calculation to finishes the initialization
        ##########################################################################################
        # Detector effects
        if 'pedestal' in geom:
            self._pedestal = xp.asarray(geom['pedestal'], dtype=xp.float64)
        else:
            self._pedestal = xp.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel rms' in geom:
            self._pixel_rms = xp.asarray(geom['pixel rms'], dtype=xp.float64)
        else:
            self._pixel_rms = xp.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel bkgd' in geom:
            self._pixel_bkgd = xp.asarray(geom['pixel bkgd'], dtype=xp.float64)
        else:
            self._pixel_bkgd = xp.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel status' in geom:
            self._pixel_status = xp.asarray(geom['pixel status'], dtype=xp.float64)
        else:
            self._pixel_status = xp.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel mask' in geom:
            self._pixel_mask = xp.asarray(geom['pixel mask'], dtype=xp.float64)
        else:
            self._pixel_mask = xp.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel gain' in geom:
            self._pixel_gain = xp.asarray(geom['pixel gain'], dtype=xp.float64)
        else:
            self._pixel_gain = xp.ones(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
