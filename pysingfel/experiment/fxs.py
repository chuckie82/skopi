import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pysingfel.geometry as psg
import pysingfel as ps
from pysingfel.particlePlacement import *

from .base import Experiment


class FXSExperiment(Experiment):
    def __init__(self, det, beam, particles, n_part_per_shot, ratios=None):
        super(FXSExperiment, self).__init__(det, beam, particles)
        self.n_part_per_shot = n_part_per_shot
        self.particles = particles
        if ratios is None:
            ratios = np.ones(len(particles))
        ratios = np.array(ratios)
        if np.any(ratios < 0):
            raise ValueError("Ratios need to be positive.")
        if len(ratios) != self.n_particle_kinds:
            raise ValueError("Need as many ratios as particles.")
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
        particle_dict = {self.particles[i]: n_particles for i, n_particles in enumerate(particle_distribution)}
        part_states, part_positions = distribute_particles(particle_dict, self.beam.get_focus()[0]/2, jet_radius=1e-4, gamma=0.5)
        print("part_positions =", part_positions)
        part_states = np.array(part_states)
        for i in range(self.n_particle_kinds):
            n_particles = particle_distribution[i]
            orientations = psg.get_random_quat(n_particles)
            positions = part_positions[part_states == self.particles[i]]
            particle_groups.append((positions, orientations))
        return particle_groups
