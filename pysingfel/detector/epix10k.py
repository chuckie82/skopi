import re
import six
import numpy as np
from pysingfel.util import deprecated
from .autoranging import AutoRangingDetector

class Epix10kDetector(AutoRangingDetector):
    def __init__(self, cameraConfig, *args, **kwargs):
        super(Epix10kDetector, self).__init__(cameraConfig=cameraConfig, *args, **kwargs)
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

    def updateFlatGains(self, flatGains):  ## not sure this makes sense in the residual world
        self.gains = self.matricize(flatGains)
        raise Exception

    def updateFlatOffsets(self, flatOffsets):
        self.offsets = self.matricize(flatOffsets)

    def updateFlatSwitchPoints(self, flatSwitchPoints):
        self.switchPoints = self.matricize(flatSwitchPoints)

    def updateSwitchPoints(self, switchPoints):
        self.switchPoints = switchPoints

    def setupMatrices(self):
        self.residualGains = self.matricize(self.gains, self.gainErrors, None)/self.matricize(self.gains, None, None) # Q: ADU/keV?
        self.offsets = self.matricize(self.offsets)
        self.switchPoints = self.matricize(self.switchPoints, None, self.switchPointVariations)

    def matricize(self, array, relativeSmear=None, absoluteSmear=None): # better to separate out smearing function
        base = np.ones((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
        tmp = []
        [tmp.append(array[i]*base) for i in range(self.nRanges)]
        if relativeSmear is not None: ## should probably be by range, handle array or scalar
            if not np.isscalar(relativeSmear):
                print("temp check in relativeSmear array handler")
                if len(relativeSmear)==self.nRanges:
                    for n in range(self.nRanges): ## lazy and unidiomatic
                        smears = 1 + (np.random.normal(size=self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))).clip(-3,3)*relativeSmear[n]
                        ## clip to eliminate unfortunate tails, e.g. negative gain 
                        tmp[n] *= smears
                else:
                    raise Exception
            else:
                smears = 1 + np.random.normal(size=self.nRanges*self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))*relativeSmear
                tmp += smears 
        if absoluteSmear is not None: ## should probably be by range, handle array or scalar
            if not np.isscalar(absoluteSmear):
                print("temp check in absoluteSmear array handler")
                if len(absoluteSmear)==self.nRanges:
                    for n in range(self.nRanges): ## lazy and unidiomatic
                        smears = (np.random.random(self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))-0.5)*absoluteSmear[n] ## shifts 0, 1 to -0.5, 0.5 and scales
                        tmp[n] += smears
                else:
                    raise Exception
            else:
                smears = (np.random.random(self.nRanges*self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))-0.5)*absoluteSmear ## shifts 0, 1 to -0.5, 0.5 and scales
                tmp += smears
        return np.array(tmp)

    def setGains(self, gains):
        if gains.shape != (self.nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]):
            print("gain problem,", gains.shape, "!=", (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
            raise
        self.gains = gains

    def setOffsets(self, offsets):
        print("foo", 1/0)
        

    def _get_cbase(self):
        """Get detector calibration base object.
        Psana 1 only.
        """
        from PSCalib.CalibParsBaseEpix10kaV1 import CalibParsBaseEpix10kaV1
        return CalibParsBaseEpix10kaV1()
