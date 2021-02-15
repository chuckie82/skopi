import re
import six
import numpy as np

from skopi.util import deprecated
from .autoranging import AutoRangingDetector

class Epix10kDetector(AutoRangingDetector):
    def __init__(self, cameraConfig, *args, **kwargs):
        super(Epix10kDetector, self).__init__(*args, cameraConfig=cameraConfig, **kwargs)
        cameraConfigs = [None, 
                         "highLow", "mediumLow", ## autoranging ePix10k
                         "fixedHigh", "fixedMedium", "fixedLow"] ## fixed
        self.nBits = 14
        self.maxEnergy = 10000*8 ## keV
        self.nRanges = 2
        if self.cameraConfig in ["mediumLow", "fixedMedium", "fixedLow"]:
            self.switchPoints = [300*8, self.maxEnergy*666] ## keV
            if self.cameraConfig == "fixedMedium":
                tmp = self.switchPoints[0]
                self.switchPoints[0] = self.maxEnergy*666
                self.maxEnergy = tmp
            if self.cameraConfig == "fixedLow":
                self.switchPoints[0] = -666
            self.switchPointVariations = [4.*8, 0.] ## keV, placeholder
            self.switchPointWidths = [4.*8, 0.] ## keV, placeholder
            self.offsets = [0, 2100] ## ADU; placeholder
            self.gains = [4.1, 0.1] ## ADU/keV
            self.gainErrors = [0.01, 0.015] ## dimensionless, placeholder
            self.nonLinearity = [0.01/2400, 0.017/self.maxEnergy] ## 1/keV, placeholder
        else:
            self.switchPoints = [100*8, self.maxEnergy*666] ## keV
            if self.cameraConfig == "fixedHigh":
                tmp = self.switchPoints[0]
                self.switchPoints[0] = self.maxEnergy*666
                self.maxEnergy = tmp
            self.switchPointVariations = [2.*8, 0.] ## keV, placeholder
            self.switchPointWidths = [2.*8, 0.] ## keV, placeholder
            self.offsets = [0, 1100] ## ADU; placeholder
            self.gains = [12.5, 0.1] ## ADU/keV
            self.gainErrors = [0.01, 0.015] ## dimensionless, placeholder
            self.nonLinearity = [0.01/800, 0.017/self.maxEnergy] ## 1/keV, placeholder

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
        from PSCalib.CalibParsBaseEpix10kaV1 import CalibParsBaseEpix10kaV1
        return CalibParsBaseEpix10kaV1()
