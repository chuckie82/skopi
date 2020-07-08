from pysingfel.util import deprecated
from pysingfel.detector.epix10k import Epix10kDetector
from pysingfel.detector.jungfrau import JungfrauDetector

class BuildAutoRangeFrames(object):
    def __init__(self, detector, I0width, I0min=0, I0max=300000, field=None):
        self.det = detector
        self.panels = self.panel_num
        self.rows = self.panel_pixel_num_x[0]
        self.cols = self.panel_pixel_num_y[0]
        self.frame = np.zeros((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]))
        self.gainBits = np.zeros((self.panel_num, self.panel_pixel_num_x[0], self.panel_pixel_num_y[0]), dtype=int)
    
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
            print("flux in pixel exceeds limit, giving up") # we should set frame[i, j] = maxEnergy+1
            raise Exception
        nonLinearity = (energy-self.det.minE[i])**2*self.det.nonLinearity[i]
        pixelVal = energy*self.det.residualGains[i, :, :, :]+self.det.offsets[i, :, :, :]+nonLinearity
        self.frame = pixelVal
        self.frame[np.where(self.frame > self.det.maxEnergy)] = self.det.maxEnergy
        self.gainBits[:, :, :] = i

    def getFrame(self):
        return self.frame

    def getGainBits(self):
        return self.gainBits
