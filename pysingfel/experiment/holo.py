import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pysingfel.geometry as psg
import pysingfel as ps
from pysingfel.particlePlacement import *

from .base import Experiment


class HOLOExperiment(Experiment):
    def __init__(self, det, beam, reference, particles, n_ref_per_shot, n_part_per_shot, ref_ratios=None, part_ratios=None):
        super(HOLOExperiment, self).__init__(det, beam, particles)
        self.n_ref_per_shot = n_ref_per_shot
        self.n_part_per_shot = n_part_per_shot
        self.reference = reference
        self.particles = particles
        self.n_reference_kinds = len(reference)

        if ref_ratios is None:
            ref_ratios = np.ones(len(reference))
        ref_ratios = np.array(ref_ratios)
        if np.any(ref_ratios < 0):
            raise ValueError("Ratios need to be positive.")
        if len(ref_ratios) != self.n_reference_kinds:
            raise ValueError("Need as many ratios as reference cluster.")
        ref_ratios /= ref_ratios.sum()  # Normalize to 1
        self.ref_ratios = ref_ratios

        if part_ratios is None:
            part_ratios = np.ones(len(particles))
        part_ratios = np.array(part_ratios)
        if np.any(part_ratios < 0):
            raise ValueError("Ratios need to be positive.")
        if len(part_ratios) != self.n_particle_kinds:
            raise ValueError("Need as many ratios as particles.")
        part_ratios /= part_ratios.sum()  # Normalize to 1
        self.part_ratios = part_ratios


    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        particle_groups = []
        if self.n_ref_per_shot >= self.n_part_per_shot:
            # reference cluster
            reference_distribution = np.random.multinomial(
                self.n_ref_per_shot, self.ref_ratios)
            ref_dict = {self.reference[i]: n_reference for i, n_reference in enumerate(reference_distribution)}
            ref_states, ref_positions = distribute_particles(ref_dict, self.beam.get_focus()[0]/2, jet_radius=1e-4, gamma=1.)
            ref_states = np.array(ref_states)
            for i in range(self.n_reference_kinds):
                n_reference = reference_distribution[i]
                ref_orientations = psg.get_random_quat(n_reference)
                ref_positions = ref_positions[ref_states == self.reference[i]]
            # particle cluster
            particle_distribution = np.random.multinomial(
                self.n_part_per_shot, self.part_ratios)
            part_dict = {self.particles[i]: n_particles for i, n_particles in enumerate(particle_distribution)}
            part_states, part_positions = distribute_particles(ref_dict, self.beam.get_focus()[0]/2, jet_radius=1e-4, gamma=1.)
            part_positions[:,1] = ref_positions[:part_positions.shape[0],1]
            part_states = np.array(part_states)
            for i in range(self.n_particle_kinds):
                n_particles = particle_distribution[i]
                part_orientations = psg.get_random_quat(n_particles)
                part_positions = part_positions[part_states == self.particles[i]]
        else:
            # particle cluster
            particle_distribution = np.random.multinomial(
                self.n_part_per_shot, self.part_ratios)
            part_dict = {self.particles[i]: n_particles for i, n_particles in enumerate(particle_distribution)}
            part_states, part_positions = distribute_particles(ref_dict, self.beam.get_focus()[0]/2, jet_radius=1e-4, gamma=1.)
            part_states = np.array(part_states)
            for i in range(self.n_particle_kinds):
                n_particles = particle_distribution[i]
                part_orientations = psg.get_random_quat(n_particles)
                part_positions = part_positions[part_states == self.particles[i]]
            # reference cluster
            reference_distribution = np.random.multinomial(
                self.n_ref_per_shot, self.ref_ratios)
            ref_dict = {self.reference[i]: n_reference for i, n_reference in enumerate(reference_distribution)}
            ref_states, reference_positions = distribute_particles(ref_dict, self.beam.get_focus()[0]/2, jet_radius=1e-4, gamma=1.)
            ref_positions[:,1] = part_positions[:ref_positions.shape[0],1]
            ref_states = np.array(reference_states)
            for i in range(self.n_reference_kinds):
                n_reference = reference_distribution[i]
                ref_orientations = psg.get_random_quat(n_reference)
                ref_positions = ref_positions[ref_states == self.reference[i]]
        positions = np.concatenate((ref_positions,part_positions), axis=0)
        orientations = np.concatenate((ref_orientations,part_orientations), axis=0)
        particle_groups.append((positions, orientations))
        return particle_groups
