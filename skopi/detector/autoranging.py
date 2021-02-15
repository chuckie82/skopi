import re
import six
import numpy as np
from skopi.util import deprecated
from .lcls import LCLSDetector

class AutoRangingDetector(LCLSDetector):
    def __init__(self, cameraConfig, *args, **kwargs):
        super(AutoRangingDetector, self).__init__(*args, **kwargs)
        self.cameraConfig = cameraConfig

    def updateFlatGains(self, flatGains):  ## not sure this makes sense in the residual world
        self.gains = self.matricize(flatGains)

    def updateFlatOffsets(self, flatOffsets):
        self.offsets = self.matricize(flatOffsets)

    def updateFlatSwitchPoints(self, flatSwitchPoints):
        self.switchPoints = self.matricize(flatSwitchPoints)

    def updateSwitchPoints(self, switchPoints):
        self.switchPoints = switchPoints

    def setupMatrices(self):
        self.residualGains = self.smear_rel(self.matricize(self.gains), self.gainErrors)/self.matricize(self.gains) # unitless
        self.offsets = self.matricize(self.offsets)
        self.switchPoints = self.smear_abs(self.matricize(self.switchPoints), self.switchPointVariations)

    def matricize(self, array):
        base = np.ones((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
        return np.array([ array[i]*base for i in range(self.nRanges)])

    def smear_rel(self, matrices, relativeSmear):
        if not np.isscalar(relativeSmear):
            if len(relativeSmear)==self.nRanges:
                for n in range(self.nRanges): ## lazy and unidiomatic
                    smears = 1 + (np.random.normal(size=self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))).clip(-3,3)*relativeSmear[n]
                    ## clip to eliminate unfortunate tails, e.g. negative gain 
                    matrices[n] *= smears
            else:
                raise Exception("nRanges not matched")
        else:
            smears = 1 + np.random.normal(size=self.nRanges*self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))*relativeSmear
            matrices += smears
        return matrices

    def smear_abs(self, matrices, absoluteSmear):
        if not np.isscalar(absoluteSmear):
            if len(absoluteSmear)==self.nRanges:
                for n in range(self.nRanges): ## lazy and unidiomatic
                    smears = (np.random.random(self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))-0.5)*absoluteSmear[n] ## shifts 0, 1 to -0.5, 0.5 and scales
                    matrices[n] += smears
            else:
                raise Exception("nRanges not matched")
        else:
            smears = (np.random.random(self.nRanges*self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))-0.5)*absoluteSmear ## shifts 0, 1 to -0.5, 0.5 and scales
            matrices += smears
        return matrices

    def setGains(self, gains):
        if gains.shape != (self.nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]):
            raise Exception("gain dimension mismatch")
        self.gains = gains
