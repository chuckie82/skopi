import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

from pysingfel import ReciprocalDetector
from pysingfel.util import xp, asnumpy


class Visualizer(object):
    def __init__(self, experiment, diffraction_rings=None, log_scale=False):
        self.experiment = experiment
        self.center = self.experiment.det.geometry.point_coord_indexes((0, 0))
        # ! Wavenumber definition != beam's.
        self.wavenumber = np.linalg.norm(self.experiment.beam.get_wavevector())
        self.distance = self.experiment.det.distance

        pixel_width = asnumpy(self.experiment.det.pixel_width)
        # Cupy doesn't have median yet.
        self.pix_width = np.median(pixel_width)

        recidet = ReciprocalDetector(self.experiment.det,
                                     self.experiment.beam)
        self.q_max = asnumpy(np.min((  # Max inscribed radius
            xp.max(recidet.pixel_position_reciprocal[..., 1]),
            -xp.min(recidet.pixel_position_reciprocal[..., 1]),
            xp.max(recidet.pixel_position_reciprocal[..., 0]),
            -xp.min(recidet.pixel_position_reciprocal[..., 0]))))

        self._auto_rings = False
        if diffraction_rings is not None:
            if diffraction_rings.lower() == "auto":
                self._auto_rings = True
            else:
                raise ValueError(
                    "Unrecognized value '{}' for argument diffraction_rings"
                    "".format(diffraction_rings))

        if log_scale:
            self._norm = LogNorm()
        else:
            self._norm = None

    def imshow(self, img):
        img = asnumpy(img)
        plt.imshow(img, norm=self._norm)
        plt.colorbar()
        plt.xlabel('Y')
        plt.ylabel('X')

        if self._auto_rings:
            self.add_diffraction_rings()

    def add_diffraction_ring(self, q):
        """Add a circle on assembled image.

        Takes a reciprocal distance (m-1) for the corresponding radius.
        Call after plt.imshow but before plt.legend.
        """
        # Radius
        s = q/(2*self.wavenumber)
        r = self.distance * 2 * s * np.sqrt(1-s**2) / (1-2*s**2)
        pix_rad = r / self.pix_width

        label = '{:.2g} A'.format((1/q)*1e+10)
        ncenter = self.center[::-1]  # Swap x & y to match coordinate system
        plt.gca().add_patch(
            Circle(ncenter, pix_rad, edgecolor='red', fill=False))
        plt.gca().annotate(label, xy=(ncenter[0], ncenter[1]+pix_rad),
                           ha="center", va="bottom", color="red")

    def add_diffraction_rings(self):
        """Add diffraction rings on assembled image.

        Attemp to find relevant values by itself.
        Call after plt.imshow but before plt.legend.
        """
        q_start = self.q_max * 0.99
        steps = (1, 2, 3, 4, 5)
        i_max = steps[-1]
        for i in steps:
            self.add_diffraction_ring(i*q_start/i_max)

class ShowMasks(object):
    def __init__(self, particle):
        self.particle = particle
        if self.particle.mesh is None:
            print('Masks not defined...')
        else:
            self.plot(np.floor(self.particle.mesh.shape[0]/2).astype('int'))

    def plot(self, islice):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6,9), sharex=True,  sharey=True, dpi=180)

        axes[0,0].set_title('Solute mask')
        axes[0,0].set_ylabel('YZ central slice')
        axes[0,0].imshow(self.particle.solute_mask[islice,...]*1, cmap='Greys_r')
        axes[1,0].set_ylabel('XZ central slice')
        axes[1,0].imshow(self.particle.solute_mask[:,islice,:]*1, cmap='Greys_r')
        axes[2,0].set_xlabel('voxel index')
        axes[2,0].set_ylabel('XY central slice')
        axes[2,0].imshow(self.particle.solute_mask[:,:,islice]*1, cmap='Greys_r')

        axes[0,1].set_title('Solvent mask')
        axes[0,1].imshow(self.particle.solvent_mask[islice,...]*1, cmap='Blues')
        axes[1,1].imshow(self.particle.solvent_mask[:,islice,:]*1, cmap='Blues')
        axes[2,1].set_xlabel('voxel index')
        axes[2,1].imshow(self.particle.solvent_mask[:,:,islice]*1, cmap='Blues')

        plt.tight_layout()
        plt.show()

