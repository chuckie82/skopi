import matplotlib.pyplot as plt
from skopi.util import xp

class SAXS():
    def __init__(self, particle, N, resmax):
        self.particle = particle
        self.N        = N
        self.qmax     = 1/resmax
        self.hkl      = self.define_hkl()
        self.qs, self.saxs = self.compute()
    
    def define_hkl(self):
        phi = xp.arccos(1-2*xp.random.rand(self.N))
        theta = xp.random.rand(self.N) * 2 * xp.pi
        q = xp.random.rand(self.N) * self.qmax
        h = q * xp.cos(theta) * xp.sin(phi)
        k = q * xp.sin(theta) * xp.sin(phi)
        l = q * xp.cos(phi)
        hkl = xp.stack((h, k, l), axis=-1)
        return hkl
    
    def compute(self):
        import skopi.gpu as gpu
        stack = gpu.calculate_diffraction_pattern_gpu(self.hkl, 
                                                      self.particle, 
                                                      return_type="intensity")
        dist = xp.linalg.norm(self.hkl, axis=-1)
        bins = xp.rint(dist/1e7).astype(xp.int)
        saxs_weights = xp.bincount(bins)
        saxs_acc = xp.bincount(bins, weights=stack)
        saxs = saxs_acc / saxs_weights
        qaccs = xp.bincount(bins, weights=dist)
        qs = qaccs / saxs_weights
        return qs, saxs
    
    def plot(self):
        plt.yscale('log')
        plt.xlim(0,self.qmax/10**10)
        plt.xlabel('q (inverse Angstroem)')
        plt.ylabel('logI')
        plt.plot(self.qs/10**10, self.saxs)
        plt.show()
