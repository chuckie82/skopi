import numpy as np
import pysingfel.geometry as psg

from .base import Experiment


class SPIExperiment(Experiment):
    def __init__(self, det, beam, particle, orientations=None):
        super(SPIExperiment, self).__init__(det, beam, [particle])
        self.set_orientations(orientations)

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.

        In the SPI case, it is only one group of one particle.
        """
        orientations = self.get_next_orientation()
        positions = np.array([[0., 0., 0.]])
        particle_groups = [  # For each particle kind
            (positions, orientations)
        ]
        return particle_groups

    def get_next_orientation(self):
        """
        Return the next orientation.
        """
        if self._orientations is None:
            return psg.get_random_quat(1)
        if self._i_orientations >= len(self._orientations):
            raise StopIteration("No more orientation available.")
        orientation = self._orientations[self._i_orientations, None]
        self._i_orientations += 1
        return orientation

    def set_orientations(self, orientations):
        self._orientations = orientations
        self._i_orientations = 0
