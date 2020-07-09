import re
import six
import numpy as np

from pysingfel.util import deprecated
from .autoranging import AutoRangingDetector

class JungfrauDetector(AutoRangingDetector):
    def __init__(self, cameraConfig, *args, **kwargs):
        super(JungfrauDetector, self).__init__(cameraConfig=cameraConfig, *args, **kwargs)

        cameraConfigs = [None, 
                         "highLow", "mediumLow", ## autoranging ePix10k
                         "fixedHigh", "fixedMedium", "fixedLow"] ## fixed

        self.nBits = 14
        self.maxEnergy = 15000*8 ##keV
        self.nRanges = 3
        self.switchPoints = [40*8, 1500*8, self.maxEnergy*666] ## keV
        self.switchPointVariations = [1.*8, 10.*8, 0.] ## keV, placeholder
        self.switchPointWidths = [1.*8, 10.*8, 0.] ## keV, placeholder
        self.offsets = [0, 2100, 1900] ## ADU; placeholder
        self.gains = [40., 1., 0.1] ## ADU/keV
        self.gainErrors = [0.01, 0.015, 0.02] ## dimensionless, placeholder
        self.nonLinearity = [0.01/320, 0.015/12000, 0.017/self.maxEnergy] ## 1/keV, placeholder

        self.minE = [0.] + self.switchPoints[0:self.nRanges-1] ## keV
        ## this is for adding nonlinearity
        ## calculated with respect to start of range
        self.nonLinearity = np.array(self.nonLinearity)
        self.nonLinearity *= 0. ## comment out to get nonlinearity

        self.setupMatrices()
        ## the idea is to allow us to use a scalar to describe the 2d array
        ## corresponding to a particular characteristic
        ## or to smear a 2d array
        ## or one might use a numpy array to set the 2d array

    def _get_cbase(self):
        """Get detector calibration base object.

        Psana 1 only.
        """
        from PSCalib.CalibParsBaseJungfrauV1 import CalibParsBaseJungfrauV1
        return CalibParsBaseJungfrauV1()


