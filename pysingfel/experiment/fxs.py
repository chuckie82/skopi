import numpy as np
import pysingfel.geometry as psg

from .base import Experiment


class FXSExperiment(Experiment):
    def __init__(self, det, beam, particles, n_part_per_shot, ratios=None):
        super(FXSExperiment, self).__init__(det, beam, particles)
        self.n_part_per_shot = n_part_per_shot
        if ratios is None:
            ratios = np.ones(len(particles))
        ratios = np.array(ratios)
        if np.any(ratios < 0):
            raise ValueError("Ratios need to be positive.")
        ratios /= ratios.sum()  # Normalize to 1
        self.ratios = ratios

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        particle_groups = []
        particle_distribution = np.random.multinomial(
            self.n_part_per_shot, self.ratios)
        for i in range(self.n_particle_kinds):
            n_particles = particle_distribution[i]
            orientations = psg.get_random_quat(n_particles)
            # TODO: use realistic positions (Iris?)
            positions = np.array(  # Currently all in center.
                [[0., 0., 0.]] * n_particles)
            particle_groups.append((positions, orientations))
        return particle_groups
