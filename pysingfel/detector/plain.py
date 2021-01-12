import numpy as np
import os
import sys

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc

from .base import DetectorBase


class PlainDetector(DetectorBase):
    """
    This object constructs a detector based on the .geom file.
    """

    def __init__(self, geom, beam):
        """
        Define parameters.
        :param geom: The geometry file that can be used to initialize the object.
        :param beam: The beam object.
        """
        super(PlainDetector, self).__init__()
        self.initialize(geom=geom, beam=beam)

    def initialize(self, geom, beam=None):
        """
        Initialize the detector with the user-defined geometry file (and perhaps
        self.initialize_pixels_with_beam).

        :param geom: The path of the .geom file.
        :param beam: The beam object.
        :return: None
        """
        ###########################################################################################
        # Initialize the geometry configuration
        ###########################################################################################
        geom = pu.read_geomfile(geom)
        self.geometry = geom

        # Set parameters
        self.panel_num = 1

        # Extract info
        self.panel_pixel_num_x = int(geom['pixel number x'])
        self.panel_pixel_num_y = int(geom['pixel number y'])
        self.pixel_num_total = np.array([self.panel_pixel_num_x * self.panel_pixel_num_y, ])
        self._distance = geom['distance']

        self.pixel_width = np.ones((self.panel_num,
                                    self.panel_pixel_num_x, self.panel_pixel_num_y)) * geom[
                               'pixel size x']
        self.pixel_height = np.ones((self.panel_num,
                                     self.panel_pixel_num_x, self.panel_pixel_num_y)) * geom[
                                'pixel size y']
        self.pixel_area = np.multiply(self.pixel_height, self.pixel_width)

        # Calculate real space position
        self.pixel_position = np.zeros(
            (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y, 3))
        # z direction position
        self.pixel_position[0, ::, ::, 2] += self.distance

        # x,y direction position
        total_length_x = (self.panel_pixel_num_x - 1) * geom[ 'pixel size x']
        total_length_y = (self.panel_pixel_num_y - 1) * geom[ 'pixel size y']

        x_coordinate_temp = np.linspace(-total_length_x / 2, total_length_x / 2,
                                        num=self.panel_pixel_num_x,
                                        endpoint=True)
        y_coordinate_temp = np.linspace(-total_length_y / 2, total_length_y / 2,
                                        num=self.panel_pixel_num_y,
                                        endpoint=True)
        mesh_temp = np.meshgrid(x_coordinate_temp, y_coordinate_temp)

        self.pixel_position[0, ::, ::, 0] = mesh_temp[0][::, ::]
        self.pixel_position[0, ::, ::, 1] = mesh_temp[1][::, ::]

        # Calculate the index map for the image
        mesh_temp = np.meshgrid(np.arange(self.panel_pixel_num_x),
                                np.arange(self.panel_pixel_num_y))
        p_map_x = np.stack((mesh_temp[0],))
        p_map_y = np.stack((mesh_temp[1],))
        self.pixel_index_map = np.stack((p_map_x, p_map_y), axis=-1)
        self.detector_pixel_num_x = self.panel_pixel_num_x
        self.detector_pixel_num_y = self.panel_pixel_num_y

        ##########################################################################################
        # Initialize the pixel effects
        ##########################################################################################
        # Initialize the detector effect parameters
        self._pedestal = np.zeros((self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
        self._pixel_rms = np.zeros((self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
        self._pixel_bkgd = np.zeros((self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
        self._pixel_status = np.zeros(
            (self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
        self._pixel_mask = np.zeros((self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
        self._pixel_gain = np.ones((self.panel_num, self.panel_pixel_num_x, self.panel_pixel_num_y))

        # Initialize the pixel effects
        self.initialize_pixels_with_beam(beam=beam)

    def assemble_image_stack(self, image_stack):
        """
        Assemble the image stack into a 2D diffraction pattern.
        For this specific object, since it only has one panel, the result is to remove the first
        dimension.

        :param image_stack: The [1, num_x, num_y] numpy array.
        :return: The [num_x, num_y] numpy array.
        """
        return np.reshape(image_stack, (self.panel_pixel_num_x, self.panel_pixel_num_y))

    def assemble_image_stack_batch(self, image_stack_batch):
        """
        Assemble the image stack batch into a stack of 2D diffraction patterns.
        For this specific object, since it has only one panel, the result is a simple reshape.

        :param image_stack_batch: The [stack_num, 1, num_x, num_y] numpy array
        :return: The [stack_num, num_x, num_y] numpy array
        """
        stack_num = image_stack_batch.shape[0]
        return np.reshape(image_stack_batch,
                          (stack_num, self.panel_pixel_num_x, self.panel_pixel_num_y))
