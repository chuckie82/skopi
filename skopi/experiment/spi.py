import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skopi.geometry as psg
import skopi as ps
from skopi.particlePlacement import *

from .base import Experiment


class SPIExperiment(Experiment):
    """
    Class for SPI experiment.
    """

    def __init__(self, det, beam, jet_radius, particles, n_part_per_shot, sticking=True, ratios=None):
        """
        Initialize a SPI experiment.
        
        :param det: The detector object.
        :param beam: The beam object.
        :param jet_radius: The radius of the gas jet.
        :param particles: The particle objects.
        :param n_part_per_shot: The number of photons per pulse/shot.
        :param sticking: If multiple particles, whether aggregated (default) or disperse.
        :param ratios: The ratios of the particles.
        """
        super(SPIExperiment, self).__init__(det, beam, particles)
        self.jet_radius = jet_radius
        self.particles = particles
        self.n_part_per_shot = n_part_per_shot
        self.sticking = sticking

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
        part_states, part_positions = distribute_particles(particle_dict, self.beam.get_focus()[0]/2, self.jet_radius, self.sticking)
        part_states = np.array(part_states)
        for i in range(self.n_particle_kinds):
            n_particles = particle_distribution[i]
            orientations = psg.get_random_quat(n_particles)
            positions = part_positions[part_states == self.particles[i]]
            particle_groups.append((positions, orientations))
        return particle_groups

    def add_fluence_jitter(self):
        """
        Return fluence_jitter

        The fluence jitters from run to run as the focus size
        and position changes with photo energy and bandwidth.
        """
        sample_state = self.generate_new_sample_state()
        positions = sample_state[0][0]
        orientations = sample_state[0][1]
        dist_to_focus = np.sqrt(positions[0][0]**2+positions[0][1]**2+positions[0][2]**2)
        fluence_jitter = np.exp(-dist_to_focus**2/(2*(self.beam.get_focus()[0]/2)**2+self.jet_radius**2))
        return fluence_jitter
