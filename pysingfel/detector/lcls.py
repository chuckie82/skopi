import numpy as np
import os
import six
import sys

# support LCLSI-py2, LCLSI-py3 and LCLSII-py3
try:
    if six.PY2:
        from PSCalib.GenericCalibPars import GenericCalibPars
        from PSCalib.GeometryAccess import GeometryAccess
    else:
        from psana.pscalib.geometry.GeometryAccess import GeometryAccess
        from psana.pscalib.calib.MDBWebUtils import calib_constants
        
except:
    print("Psana functionality is not available.")

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc
from pysingfel.util import xp, asnumpy

from .base import DetectorBase


class LCLSDetector(DetectorBase):
    """
    Class for lcls detectors.
    """

    def __init__(self, geom, beam=None, run_num=0, cframe=0):
        """
        Initialize a pnccd detector.

        :param geom: The path to the geometry .data file.
        :param beam: The beam object.
        :param run_num: The run_num containing the background, rms and gain and the other pixel
        pixel properties.
        :param cframe: The desired coordinate frame, 0 for psana and 1 for lab conventions. The
        default (psana) matches the convention of non-LCLS detectors. Lab frame yields the transpose.
        """
        super(LCLSDetector, self).__init__()

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

        self.initialize(geom=geom, run_num=run_num, cframe=cframe)
        # Initialize the pixel effects, enforcing detector distance to be positive
        if self.distance < 0:
            self.distance *= -1
        self.initialize_pixels_with_beam(beam=beam)

    def initialize(self, geom, run_num=0, cframe=0):
        """
        Initialize the detector as pnccd
        :param geom: The pnccd .data file which characterize the geometry profile.
        :param run_num: The run_num containing the background, rms and gain and the other
                        pixel pixel properties.
        :param cframe: The desired coordinate frame, 0 for psana and 1 for lab conventions.
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
        self.geometry = GeometryAccess(geom, cframe=cframe)
        self.run_num = run_num

        # Set coordinate in real space (convert to m)
        temp = [xp.asarray(t) * 1e-6 for t in self.geometry.get_pixel_coords(cframe=cframe)]
        temp_index = [xp.asarray(t)
                      for t in self.geometry.get_pixel_coord_indexes(cframe=cframe)]

        self.panel_num = np.prod(temp[0].shape[:-2])
        self._distance = float(temp[2].mean())

        self._shape = (self.panel_num, temp[0].shape[-2], temp[0].shape[-1])
        self.pixel_position = xp.zeros(self._shape + (3,))
        self.pixel_index_map = xp.zeros(self._shape + (2,))

        for n in range(3):
            self.pixel_position[..., n] = temp[n].reshape(self._shape)
        for n in range(2):
            self.pixel_index_map[..., n] = temp_index[n].reshape(self._shape)

        self.pixel_index_map = self.pixel_index_map.astype(xp.int64)

        # Get the range of the pixel index
        self.detector_pixel_num_x = asnumpy(
            xp.max(self.pixel_index_map[..., 0]) + 1)
        self.detector_pixel_num_y = asnumpy(
            xp.max(self.pixel_index_map[..., 1]) + 1)

        self.panel_pixel_num_x = np.array([self.pixel_index_map.shape[1], ] * self.panel_num)
        self.panel_pixel_num_y = np.array([self.pixel_index_map.shape[2], ] * self.panel_num)
        self.pixel_num_total = np.sum(np.multiply(self.panel_pixel_num_x, self.panel_pixel_num_y))

        tmp = float(self.geometry.get_pixel_scale_size() * 1e-6)  # Convert to m
        self.pixel_width = xp.ones(
            (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0])) * tmp
        self.pixel_height = xp.ones(
            (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0])) * tmp

        # Calculate the pixel area
        self.pixel_area = xp.multiply(self.pixel_height, self.pixel_width)

        ###########################################################################################
        # Initialize the pixel effects
        ###########################################################################################
        # first we should parse the path
        parsed_path = geom.split('/')
        group = parsed_path[-4]
        source = parsed_path[-3]

        self._pedestals = None
        self._pixel_rms = None
        self._pixel_mask = None
        self._pixel_bkgd = None
        self._pixel_status = None
        self._pixel_gain = None

        if six.PY2:
            try:
                cbase = self._get_cbase()
                calibdir = '/'.join(parsed_path[:-4])
                pbits = 255
                gcp = GenericCalibPars(cbase, calibdir, group, source, run_num, pbits)

                self._pedestals = gcp.pedestals()
                self._pixel_rms = gcp.pixel_rms()
                self._pixel_mask = gcp.pixel_mask()
                self._pixel_bkgd = gcp.pixel_bkgd()
                self._pixel_status = gcp.pixel_status()
                self._pixel_gain = gcp.pixel_gain()
            except NotImplementedError:
                # No GenericCalibPars information.
                pass
        else:
            try:
                self.det = self._get_det_id(source)
            except NotImplementedError:
                # No GenericCalibPars information.
                self.det = None
            self.exp = parsed_path[-5]

        # Redirect the output stream
        sys.stdout = old_stdout
        # f.close()
        # os.remove('./Detector_initialization.log')

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        raise NotImplementedError()

    def _get_det_id(self, source):
        """Get detector ID form source.

        Psana 2 only.
        """
        raise NotImplementedError()

    def _get_calib_constants(self, name):
        _name = "_" + name
        attribute = getattr(self, _name)
        if six.PY3 and attribute is None and self.det is not None:
            # We haven't tried to get the calib_constant yet.
            attribute = calib_constants(
                self.det, exp=self.exp, ctype=name,
                run=self.run_num)[0]
        if attribute is None:
            # We still don't have it
            raise RuntimeError("No {} available for this detector"
                               "".format(name))
        setattr(self, _name, attribute)
        return attribute

    @property
    def pedestals(self):
        return self._get_calib_constants("pedestals")

    @property
    def pixel_rms(self):
        return self._get_calib_constants("pixel_rms")

    @property
    def pixel_mask(self):
        return self._get_calib_constants("pixel_mask")

    @property
    def pixel_bkgd(self):
        return self._get_calib_constants("pixel_bkgd")

    @property
    def pixel_status(self):
        return self._get_calib_constants("pixel_status")

    @property
    def pixel_gain(self):
        return self._get_calib_constants("pixel_gain")
