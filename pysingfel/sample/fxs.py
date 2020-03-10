import numpy as np
import pysingfel.geometry as psg

from .base import Sample


class FXSSample(Sample):
    def __init__(self, particles, n_part_of_each_per_shot):
        super(FXSSample, self).__init__(particles)
        self.n_part_of_each_per_shot = n_part_of_each_per_shot

    def generate_new_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        particle_groups = []
        for i in range(self.n_particle_kinds):
            orientations = psg.get_random_quat(
                self.n_part_of_each_per_shot)
            # TODO: use realistic positions (Iris?)
            positions = np.array(  # Currently all in center.
                [[0., 0., 0.]] * self.n_part_of_each_per_shot)
            particle_groups.append((positions, orientations))
        return particle_groups
