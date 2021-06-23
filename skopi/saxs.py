import numpy as np
import matplotlib.pyplot as plt
from skopi.util import xp

class SAXS():
    def __init__(self, particle, N, resmax):
        """
        Computing SAXS curve of a particle.

        Parameters
        ----------
        particle : Particle()
            A particle object for a pdb entry of interest.

        N : int
            Number of random HKL samples.

        resmax: float
            Maximum resolution of the SAXS curve (m).

        Examples
        --------

        >>> particle = sk.Particle()
        >>> particle.read_pdb("3iyf.pdb", ff='WK')
        >>> saxs = sk.SAXS(particle, 100000, 1e-9)
        >>> saxs.plot()
        """
        self.particle = particle
        self.N        = N
        self.qmax     = 1/resmax
        self.hkl      = self.define_hkl()
        self.qs, self.saxs = self.compute()
    
    def define_hkl(self):
        """
        Generate random reciprocal points (hkl) within qmax.
        """

        phi = xp.arccos(1-2*xp.random.rand(self.N))
        theta = xp.random.rand(self.N) * 2 * xp.pi
        q = xp.random.rand(self.N) * self.qmax
        h = q * xp.cos(theta) * xp.sin(phi)
        k = q * xp.sin(theta) * xp.sin(phi)
        l = q * xp.cos(phi)
        hkl = xp.stack((h, k, l), axis=-1)
        return hkl
    
    def compute(self):
        """
        Compute diffraction intensity of the particle at points hkl
        then bin along q.
        """

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
        if xp is np:
          return qs, saxs
        else:
          return qs.get(), saxs.get()
    
    def plot(self):
        """
        Plot SAXS curve in log scale.
        """
        qmaxAng = self.qmax*1e-10 # convert metre to Angstroem
        qsAng = self.qs*1e-10

        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_yscale('log')
        ax.set_xlim(0,qmaxAng)
        ax.set_xlabel('q (inverse Angstroem)')
        ax.set_ylabel('logI')
        ax.plot(qsAng, self.saxs)
        plt.show()
