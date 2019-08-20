import numpy as np
import os
import six
import sys

if six.PY2:
    from PSCalib.GenericCalibPars import GenericCalibPars
    from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
    from PSCalib.GeometryAccess import GeometryAccess
else:
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess
    from psana.pscalib.calib.MDBWebUtils import calib_constants

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc

from .base import DetectorBase


class PnccdDetector(DetectorBase):
    """
    Class for lcls detectors.
    """

    def __init__(self, geom, beam, run_num=0):
        """
        Initialize a pnccd detector.

        :param geom: The path to the geometry .data file.
        :param beam: The beam object.
        :param run_num: The run_num containing the background, rms and gain and the other pixel
        pixel properties.
        """
        super(PnccdDetector, self).__init__()

        # Parse the path to extract the necessary information to use psana modules
        parsed_path = geom.split('/')
        # Notify the user that the path should be as deep as the geometry profile
        if parsed_path[-2] != "geometry":
            # print parsed_path[-1]
            raise Exception(
                " Sorry, at present, the package is not very smart. Please specify " +

                "the path of the detector as deep as the geometry profile. \n " +
                "And example would be like:" +
                "/reg/d/psdm/amo/experiment_name/calib/group/source/geometry/0-end.data \n" +
                "where the '/calib/group/source/geometry/0-end.data' part is essential. \n" +
                "The address before that part is not essential and can be replaced with" +
                " your absolute address or relative address.\n"
                "The experiment_name is also essential in Python 3.")

        self.initialize(geom=geom, run_num=run_num)

        # Initialize the pixel effects
        self.initialize_pixels_with_beam(beam=beam)

    def initialize(self, geom, run_num=0):
        """
        Initialize the detector as pnccd
        :param geom: The pnccd .data file which characterize the geometry profile.
        :param run_num: The run_num containing the background, rms and gain and the other
                        pixel pixel properties.
        :return:  None
        """

        # Redirect the output stream
        old_stdout = sys.stdout
        f = six.StringIO()
        # f = open('Detector_initialization.log', 'w')
        sys.stdout = f

        ###########################################################################################
        # Initialize the geometry configuration
        ############################################################################################
        self.geometry = GeometryAccess(geom, 0o377)
        self.run_num = run_num

        # Set coordinate in real space
        temp = self.geometry.get_pixel_coords()
        temp_index = self.geometry.get_pixel_coord_indexes()

        self.panel_num = temp[0].shape[1] * temp[0].shape[2]
        self.distance = temp[2][0, 0, 0, 0, 0] * 1e-6  # Convert to m

        self.pixel_position = np.zeros((self.panel_num, temp[0].shape[3], temp[0].shape[4], 3))
        self.pixel_index_map = np.zeros((self.panel_num, temp[0].shape[3], temp[0].shape[4], 2))

        for l in range(temp[0].shape[1]):
            for m in range(temp[0].shape[2]):
                for n in range(3):
                    self.pixel_position[m + l * temp[0].shape[2], :, :, n] = temp[n][0, l, m]
                for n in range(2):
                    self.pixel_index_map[m + l * temp[0].shape[2], :, :, n] = temp_index[n][0, l, m]

        self.pixel_index_map = self.pixel_index_map.astype(np.int64)

        # Get the range of the pixel index
        self.detector_pixel_num_x = np.max(self.pixel_index_map[:, :, :, 0]) + 1
        self.detector_pixel_num_y = np.max(self.pixel_index_map[:, :, :, 1]) + 1

        self.panel_pixel_num_x = np.array([self.pixel_index_map.shape[1], ] * self.panel_num)
        self.panel_pixel_num_y = np.array([self.pixel_index_map.shape[2], ] * self.panel_num)
        self.pixel_num_total = np.sum(np.multiply(self.panel_pixel_num_x, self.panel_pixel_num_y))

        tmp = float(self.geometry.get_pixel_scale_size() * 1e-6)  # Convert to m
        self.pixel_width = np.ones(
            (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0])) * tmp
        self.pixel_height = np.ones(
            (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0])) * tmp

        # Calculate the pixel area
        self.pixel_area = np.multiply(self.pixel_height, self.pixel_width)

        ###########################################################################################
        # Initialize the pixel effects
        ###########################################################################################
        # first we should parse the path
        parsed_path = geom.split('/')
        source = parsed_path[-3]

        if six.PY2:
            cbase = CalibParsBasePnccdV1()
            calibdir = '/'.join(parsed_path[:-4])
            group = parsed_path[-4]
            pbits = 255
            gcp = GenericCalibPars(cbase, calibdir, group, source, run_num, pbits)

            self._pedestals = gcp.pedestals()
            self._pixel_rms = gcp.pixel_rms()
            self._pixel_mask = gcp.pixel_mask()
            self._pixel_bkgd = gcp.pixel_bkgd()
            self._pixel_status = gcp.pixel_status()
            self._pixel_gain = gcp.pixel_gain()
        else:
            self.det = "pnccd_000" + source[-1]
            self.exp = parsed_path[-5]

            self._pedestals = None
            self._pixel_rms = None
            self._pixel_mask = None
            self._pixel_bkgd = None
            self._pixel_status = None
            self._pixel_gain = None

        # Redirect the output stream
        sys.stdout = old_stdout
        # f.close()
        # os.remove('./Detector_initialization.log')

    @property
    def pedestals(self):
        if six.PY3 and not self._pedestals:
            self._pedestals = calib_constants(
                self.det, exp=self.exp, ctype="pedestals",
                run=self.run_num)[0]
        return self._pedestals

    @property
    def pixel_rms(self):
        if six.PY3 and not self._pixel_rms:
            self._pixel_rms = calib_constants(
                self.det, exp=self.exp, ctype="pixel_rms",
                run=self.run_num)[0]
        return self._pixel_rms

    @property
    def pixel_mask(self):
        if six.PY3 and not self._pixel_mask:
            self._pixel_mask = calib_constants(
                self.det, exp=self.exp, ctype="pixel_mask",
                run=self.run_num)[0]
        return self._pixel_mask

    @property
    def pixel_bkgd(self):
        if six.PY3 and not self._pixel_bkgd:
            self._pixel_bkgd = calib_constants(
                self.det, exp=self.exp, ctype="pixel_bkgd",
                run=self.run_num)[0]
        return self._pixel_bkgd

    @property
    def pixel_status(self):
        if six.PY3 and not self._pixel_status:
            self._pixel_status = calib_constants(
                self.det, exp=self.exp, ctype="pixel_status",
                run=self.run_num)[0]
        return self._pixel_status

    @property
    def pixel_gain(self):
        if six.PY3 and not self._pixel_gain:
            self._pixel_gain = calib_constants(
                self.det, exp=self.exp, ctype="pixel_gain",
                run=self.run_num)[0]
        return self._pixel_gain

    def assemble_image_stack(self, image_stack):
        """
        Assemble the image stack into a 2D diffraction pattern.
        For this specific object, since it only has one panel, the result is to remove the
        first dimension.

        :param image_stack: The [1, num_x, num_y] numpy array.
        :return: The [num_x, num_y] numpy array.
        """
        # construct the image holder:
        image = np.zeros((self.detector_pixel_num_x, self.detector_pixel_num_y))
        for l in range(self.panel_num):
            image[self.pixel_index_map[l, :, :, 0],
                  self.pixel_index_map[l, :, :, 1]] = image_stack[l, :, :]

        return image

    def assemble_image_stack_batch(self, image_stack_batch):
        """
        Assemble the image stack batch into a stack of 2D diffraction patterns.
        For this specific object, since it has only one panel, the result is a simple reshape.

        :param image_stack_batch: The [stack_num, 1, num_x, num_y] numpy array
        :return: The [stack_num, num_x, num_y] numpy array
        """
        stack_num = image_stack_batch.shape[0]

        # construct the image holder:
        image = np.zeros((stack_num, self.detector_pixel_num_x, self.detector_pixel_num_y))
        for l in range(self.panel_num):
            idx_map_1 = self.pixel_index_map[l, :, :, 0]
            idx_map_2 = self.pixel_index_map[l, :, :, 1]
            image[:, idx_map_1, idx_map_2] = image_stack_batch[:, l]

        return image
