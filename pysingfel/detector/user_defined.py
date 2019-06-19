import numpy as np
import os
import sys

from PSCalib.GenericCalibPars import GenericCalibPars
from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc

from .base import DetectorBase


class UserDefinedDetector(DetectorBase):
    """
    Class for user defined detector. The user has to provide the necessary information
    with a dictionary with proper entries to use this class.
    """

    def __init__(self, geom, beam):
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
        ##########################################################################################
        # Extract necessary information
        ##########################################################################################

        # Define the hierarchy system. For simplicity, we only use two-layer structure.
        self.panel_num = int(geom['panel number'])

        # Define all properties the detector should have
        self.distance = float(geom['detector distance'])  # detector distance in (m)
        self.pixel_width = geom['pixel width'].astype(
            np.float64)  # [panel number, pixel num x, pixel num y]  in (m)
        self.pixel_height = geom['pixel height'].astype(
            np.float64)  # [panel number, pixel num x, pixel num y]  in (m)
        self.center_x = geom['pixel center x'].astype(
            np.float64)  # [panel number, pixel num x, pixel num y]  in (m)
        self.center_y = geom['pixel center y'].astype(
            np.float64)  # [panel number, pixel num x, pixel num y]  in (m)
        self.orientation = np.array([0, 0, 1])

        # construct the the pixel position array
        self.pixel_position = np.zeros((self.panel_num, self.panel_pixel_num_x,
                                        self.panel_pixel_num_y, 3))
        self.pixel_position[:, :, :, 2] = self.distance
        self.pixel_position[:, :, :, 0] = self.center_x
        self.pixel_position[:, :, :, 1] = self.center_y

        # Pixel map
        self.pixel_index_map = geom['pixel map'].astype(
            np.int64)  # [panel number, pixel num x, pixel num y]

        # Detector pixel number info
        self.detector_pixel_num_x = np.max(self.pixel_index_map[:, :, :, 0])
        self.detector_pixel_num_y = np.max(self.pixel_index_map[:, :, :, 1])

        # Panel pixel number info
        self.panel_pixel_num_x = self.pixel_index_map.shape[
            1]  # number of pixels in each panel in x direction
        self.panel_pixel_num_y = self.pixel_index_map.shape[
            2]  # number of pixels in each panel in y direction

        # total number of pixels (px*py)
        self.pixel_num_total = self.panel_num * self.panel_pixel_num_x * self.panel_pixel_num_y

        ###########################################################################################
        # Do necessary calculation to finishes the initialization
        ###########################################################################################
        # self.geometry currently only work for the pre-defined detectors
        self.geometry = geom

        # Calculate the pixel area
        self.pixel_area = np.multiply(self.pixel_height, self.pixel_width)

        # Get reciprocal space configurations and corrections.
        self.initialize_pixels_with_beam(beam=beam)

        ##########################################################################################
        # Do necessary calculation to finishes the initialization
        ##########################################################################################
        # Detector effects
        if 'pedestal' in geom:
            self.pedestal = geom['pedestal'].astype(np.float64)
        else:
            self.pedestal = np.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel rms' in geom:
            self.pixel_rms = geom['pixel rms'].astype(np.float64)
        else:
            self.pixel_rms = np.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel bkgd' in geom:
            self.pixel_bkgd = geom['pixel bkgd'].astype(np.float64)
        else:
            self.pixel_bkgd = np.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel status' in geom:
            self.pixel_status = geom['pixel status'].astype(np.float64)
        else:
            self.pixel_status = np.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel mask' in geom:
            self.pixel_mask = geom['pixel mask'].astype(np.float64)
        else:
            self.pixel_mask = np.zeros(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        if 'pixel gain' in geom:
            self.pixel_gain = geom['pixel gain'].astype(np.float64)
        else:
            self.pixel_gain = np.ones(
                (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
