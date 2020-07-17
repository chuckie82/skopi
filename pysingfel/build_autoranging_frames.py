from pysingfel.util import deprecated
from pysingfel.detector.epix10k import Epix10kDetector
from pysingfel.detector.jungfrau import JungfrauDetector
import numpy as np

class BuildAutoRangeFrames(object):
    def __init__(self, detector, I0width, I0min=0, I0max=300000, field=None):
        self.det = detector
        self.panels = self.det.panel_num
        self.rows = self.det.panel_pixel_num_x[0]
        self.cols = self.det.panel_pixel_num_y[0]
        self.frame = np.zeros((self.det.panel_num, self.det.panel_pixel_num_x[0], self.det.panel_pixel_num_y[0]))
        self.gainBits = np.zeros((self.det.panel_num, self.det.panel_pixel_num_x[0], self.det.panel_pixel_num_y[0]), dtype=int)

        self.I0max = I0max
        self.I0min = I0min
        self.I0saturated = 250000 # empirical value
        self.I0width = I0width
        self.field = field
    
    def getTrueI0(self):
        self.trueI0 = np.random.random()*(self.I0max-self.I0min) + self.I0min

    def getI0(self):
        self.getTrueI0()
        if self.trueI0 > self.I0saturated:
            return self.I0saturated
        self.I0 = self.trueI0 * (1.+(np.random.random()-0.5)*self.I0width)
    
    def makeFrame(self):
        self.getI0()
        energy = self.field ## keV
        detectorFried = True
        for i in range(self.det.nRanges):
            if np.any(energy[:,:,:] < (self.det.switchPoints[i, :, :, :] + self.det.switchPointWidths[i]*np.random.normal())):
                detectorFried = False
                break
        if detectorFried:
            raise Exception("flux in pixel exceeds limit, giving up")
        nonLinearity = (energy-self.det.minE[i])**2*self.det.nonLinearity[i]
        pixelVal = energy*self.det.residualGains[i][:][:][:]+self.det.offsets[i][:][:][:]+nonLinearity
        self.frame = pixelVal
        self.frame[np.where(self.frame > self.det.maxEnergy)] = self.det.maxEnergy
        self.gainBits[:, :, :] = i

    def getFrame(self):
        return self.frame

    def getGainBits(self):
        return self.gainBits
