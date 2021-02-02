import numpy as np
import os
import six
import sys

# support LCLSI-py2 and LCLSII-py3
# currently LCLSI-py3 is not supported
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
    Class for LCLS detectors.
    """

    def __init__(self, geom, beam=None, run_num=0, cframe=0):
        """
        Initialize a LCLS detector.

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
        Initialize the detector
        :param geom: The *-end.data file which characterizes the geometry profile.
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
        self.exp = parsed_path[-5]
        if self.exp == 'calib':
            self.exp = parsed_path[-6]
        self.group = parsed_path[-4]
        self.source = parsed_path[-3]

        self._pedestals = None
        self._pixel_rms = None
        self._pixel_mask = None
        self._pixel_bkgd = None
        self._pixel_status = None
        self._pixel_gain = None

        if six.PY2:
            try:
                cbase = self._get_cbase()
                self.calibdir = '/'.join(parsed_path[:-4])
                pbits = 255
                gcp = GenericCalibPars(cbase, self.calibdir, self.group, self.source, run_num, pbits)

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
                self.det = self._get_det_id(self.group)
            except NotImplementedError:
                # No GenericCalibPars information.
                self.det = None

        # Redirect the output stream
        sys.stdout = old_stdout
        # f.close()
        # os.remove('./Detector_initialization.log')

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        raise NotImplementedError()

    def _get_det_id(self, group):
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

    @pedestals.setter
    def pedestals(self, value):
        self._pedestals = value

    @pixel_rms.setter
    def pixel_rms(self, value):
        self._pixel_rms = value

    @pixel_mask.setter
    def pixel_mask(self, value):
        self._pixel_mask = value

    @pixel_bkgd.setter
    def pixel_bkgd(self, value):
        self._pixel_bkgd = value

    @pixel_status.setter
    def pixel_status(self, value):
        self._pixel_status = value
    
    @pixel_gain.setter
    def pixel_gain(self, value):
        self._pixel_gain = value

    def reset_calib(self, run_num):
        """
        Update calibration pixel effects based on new run number.
        """
        old_stdout = sys.stdout
        f = six.StringIO()
        sys.stdout = f

        self.run_num = run_num
        
        if six.PY2:
            try:
                pbits = 255
                gcp = GenericCalibPars(self._get_cbase(), self.calibdir, self.group, 
                                       self.source, self.run_num, pbits)
                self._pedestals = gcp.pedestals()
                self._pixel_rms = gcp.pixel_rms()
                self._pixel_mask = gcp.pixel_mask()
                self._pixel_bkgd = gcp.pixel_bkgd()
                self._pixel_status = gcp.pixel_status()
                self._pixel_gain = gcp.pixel_gain()
            except NotImplementedError:
                pass
        else:
            self._pedestals = calib_constants(self.det, exp=self.exp, ctype='pedestals', run=self.run_num)[0]
            self._pixel_rms = calib_constants(self.det, exp=self.exp, ctype='pixel_rms', run=self.run_num)[0]
            self._pixel_mask = calib_constants(self.det, exp=self.exp, ctype='pixel_mask', run=self.run_num)[0]
            self._pixel_bkgd = calib_constants(self.det, exp=self.exp, ctype='pixel_bkgd', run=self.run_num)[0]
            self._pixel_status = calib_constants(self.det, exp=self.exp, ctype='pixel_status', run=self.run_num)[0]
            self._pixel_gain = calib_constants(self.det, exp=self.exp, ctype='pixel_gain', run=self.run_num)[0]

        sys.stdout = old_stdout

        return

    ###########################################################################################
    # Functionality for adding dark noise
    ###########################################################################################
    def _calibrate_evt(self, evt):
        """
        Retrieve calibrated data from psana event object. Applied corrections are 
        pedestal, common mode, gain mask, gain, and pixel status mask, performed
        by the psana.Detector class.
    
        :param evt: psana event object
        :return data: calibrated image
        """    
        import psana

        # retrieve psana.Source alias
        det_type = self.__class__.__name__.split("Detector")[0].lower()
        alias = None
    
        for key in evt.keys():
            if det_type in key.alias().lower():
                alias = key.alias()
                break
            else:
                srcname = key.src()
                if srcname.__class__.__name__ == 'DetInfo':
                    if det_type in srcname.devName().lower():
                        alias = str(srcname)
                        break

        # retrieve calibrated shot
        det = psana.Detector(alias)
        return det.calib(evt)

    def _retrieve_batch_evt(self, num_shots):
        """
        Retrieve num_shots patterns from a run of the experiment.
        
        :param num_shots: number of patterns to retrieve
        :return data: array of patterns in shape (num_shots, n_pedestals, ped_x, ped_y)
        """
        # set up psana1 DataSource object
        from psana import DataSource
        ds = DataSource('exp=%s:run=%i' %(self.exp, self.run_num))
    
        # set up storage array
        if self.pedestals.ndim == 4:
            pshape = self.pedestals.shape[1:]
        else:
            pshape = self.pedestals.shape
        data = np.zeros((num_shots, pshape[0], pshape[1], pshape[2]))

        # retrieve multiple events (shots)
        counter = 0
        for num,evt in enumerate(ds.events()):
            if counter < num_shots:
                data[counter] = np.array(self._calibrate_evt(evt))
                counter += 1
            else:
                break

        # if run is shorter than num_shots, fill in remainder by linear combination
        if counter < num_shots:
            for i in range(counter, num_shots):
                indices = np.random.randint(0, high=counter, size=2)
                weights = np.random.dirichlet(np.ones(2))
                data[i] = weights[0]*data[indices[0]] + weights[1]*data[indices[1]]
            
        return data

    def _random_dark_index(self):
        """
        Return the run index of random dark run, assuming that the indices of dark
        runs can be inferred from the pedestal nomenclature.
        
        :return dark_idx: index of random dark run, -1 if no dark runs available
        """
        import glob
        
        # list of available pedestals
        pnames = glob.glob("/reg/d/psdm/%s/%s/calib/%s/%s/pedestals/*-end.data" %(self.exp[:3].upper(), self.exp, 
                                                                                  self.group, self.source))

        # add run indices from pedestals list if associated XTC files exist
        dark_indices = list()
        for pn in pnames:
            temp_str = pn.split("/")[-1]
            temp_idx = int(temp_str.split("-")[0])
            fnames = glob.glob("/reg/d/psdm/%s/%s/xtc/*-r%04d-*.xtc" %(self.exp[:3].upper(), self.exp, temp_idx))
            if len(fnames) > 0:
                dark_indices.append(temp_idx)
            
        # return random dark run or -1 if none available
        if len(dark_indices) != 0:
            return np.random.choice(np.array(dark_indices))
        else:
            return -1
            
    def add_dark_noise(self, num_shots, det_shape=True, dark_idx=None, mask_neg=True):
        """
        Retrieve calibrated images from dark runs.
    
        :param num_shots: number of calibrated dark shots to retreive
        :param det_shape: boolean, if True reassemble panels into detector's shape
        :param dark_idx: index of dark run; if None, a run number will be chosen randomly
        :param mask_neg: boolean, if True set negative-valued pixels to zero
        :return dark_data: array of calibrated dark shots with shape 
           (num_shots, det_x, det_y) if det_shape is True 
           (num_shots, n_panels, panel_x, panel_y) if det_shape is False
           None if pedestals and/or XTC files for a dark run are unavailable
        """
        if six.PY3:
            raise NotImplementedError('Currently only implemented for psana2/python3.')
            return

        # grab index of random dark run and reset calibration attributes to match
        if dark_idx == None:
            dark_idx = self._random_dark_index()
            if dark_idx == -1:
                print("Pedestals and/or XTC data are unavailable.")
                return
        self.reset_calib(dark_idx)

        # retrieve dark data
        dark_data = self._retrieve_batch_evt(num_shots)

        # floor: set negative intensities to zero
        if mask_neg:
            dark_data[dark_data<0] = 0
    
        # optionally reshape to match detector's shape
        if det_shape:
            dark_data = self.assemble_image_stack_batch(dark_data)
        
        return dark_data
