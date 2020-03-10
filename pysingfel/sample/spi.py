import numpy as np
import pysingfel.geometry as psg

from .base import Sample


class SPISample(Sample):
    def __init__(self, particle):
        super(SPISample, self).__init__([particle])

    def generate_new_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.

        In the SPI case, it is only one group of one particle.
        """
        orientations = psg.get_random_quat(1)
        positions = np.array([[0., 0., 0.]])
        particle_groups = [  # For each particle kind
            (positions, orientations)
        ]
        return particle_groups
