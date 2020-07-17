import re
import six
import numpy as np
from pysingfel.util import deprecated
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
        self.residualGains = self.matricize(self.gains, self.gainErrors, None)/self.matricize(self.gains, None, None) # Q: ADU/keV?
        self.offsets = self.matricize(self.offsets)
        self.switchPoints = self.matricize(self.switchPoints, None, self.switchPointVariations)

    def matricize(self, array, relativeSmear=None, absoluteSmear=None):
        base = np.ones((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
        tmp = []
        [tmp.append(array[i]*base) for i in range(self.nRanges)]        
        def smear(self, tmp, relativeSmear=None, absoluteSmear=None):
            if relativeSmear is not None: ## should probably be by range, handle array or scalar
                if not np.isscalar(relativeSmear):
                    print("temp check in relativeSmear array handler")
                    if len(relativeSmear)==self.nRanges:
                        for n in range(self.nRanges): ## lazy and unidiomatic
                            smears = 1 + (np.random.normal(size=self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))).clip(-3,3)*relativeSmear[n]
                            ## clip to eliminate unfortunate tails, e.g. negative gain 
                            tmp[n] *= smears
                    else:
                        raise Exception("nRanges not matched")
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
                        raise Exception("nRanges not matched")
                else:
                    smears = (np.random.random(self.nRanges*self.panel_num*self.panel_pixel_num_x[0]*self.panel_pixel_num_y[0]).reshape((nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))-0.5)*absoluteSmear ## shifts 0, 1 to -0.5, 0.5 and scales
                tmp += smears
            return tmp
        return np.array(smear(self, tmp, relativeSmear=None, absoluteSmear=None))

    def setGains(self, gains):
        if gains.shape != (self.nRanges, self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]):
            print("gain problem,", gains.shape, "!=", (self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
        self.gains = gains
