import numpy as np
import matplotlib.pyplot as plt
import pysingfel as ps
import pysingfel.gpu as gpu

class SAXS():
    def __init__(self, particle, N, resmax):
        self.particle = particle
        self.N        = N
        self.qmax     = 1/resmax
        self.hkl      = self.define_hkl()
        self.qs, self.saxs = self.compute()
    
    def define_hkl(self):
        phi = np.arccos(1-2*np.random.rand(self.N))
        theta = np.random.rand(self.N) * 2 * np.pi
        q = np.random.rand(self.N) * self.qmax
        h = q * np.cos(theta) * np.sin(phi)
        k = q * np.sin(theta) * np.sin(phi)
        l = q * np.cos(phi)
        hkl = np.stack((h, k, l), axis=-1)
        return hkl
    
    def compute(self):
        stack = gpu.calculate_diffraction_pattern_gpu(self.hkl, 
                                                         self.particle, 
                                                         return_type="intensity")
        dist = np.linalg.norm(self.hkl, axis=-1)
        bins = np.rint(dist/1e7).astype(np.int)
        saxs_weights = np.bincount(bins)
        saxs_acc = np.bincount(bins, weights=stack)
        saxs = saxs_acc / saxs_weights
        qaccs = np.bincount(bins, weights=dist)
        qs = qaccs / saxs_weights
        return qs, saxs
    
    def plot(self):
        plt.yscale('log')
        plt.xlim(0,self.qmax/10**10)
        plt.xlabel('q (inverse Angstroem)')
        plt.ylabel('logI')
        plt.plot(self.qs/10**10, self.saxs)
        plt.show()
