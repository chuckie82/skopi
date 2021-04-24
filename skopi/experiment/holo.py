import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import Counter
import skopi.geometry as psg
import skopi as ps
from skopi.particlePlacement import *

from .base import Experiment


class HOLOExperiment(Experiment):
    """
    Class for holography experiment.
    """

    def __init__(self, det, beam, reference, particles, jet_radius, ref_jet_radius,
                 ref_position=None, ref_orientation=None, part_positions=None, part_orientations=None):
        """
        Initialize a holography experiment. Here we assume a single reference
        particle and n target particles.
        
        :param det: The detector object.
        :param beam: The beam object.
        :param reference: The reference particle object.
        :param particles: The sample particle objects.
        :param jet_radius: The radius of the aerosolized jet used to deliver target particles.
        :param ref_jet_radius: The radius of the aerosolized jet used to deliver the reference particle.
        :param ref_position: The position of the reference particle object.
        :param ref_orientation: The orientation of the reference particle object.
        :param part_positions: The positions of the sample particle objects.
        :param part_orientations: The orientations of the sample particle objects.
        """
        super(HOLOExperiment, self).__init__(det, beam, reference + particles)
        self.n_target_particles = len(particles)
        self.particles = particles
        self.reference = reference
        self.jet_radius = jet_radius
        self.ref_jet_radius = ref_jet_radius
        self.set_ref_position(ref_position)
        self.set_ref_orientation(ref_orientation)
        self.set_part_positions(part_positions)        
        self.set_part_orientations(part_orientations)

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        part_orientations = self.get_next_part_orientation()
        part_positions = self.get_next_part_position()
        ref_orientation = self.get_ref_orientation()
        ref_position = self.get_ref_position()
        if ref_position is None or ref_orientation is None:
            raise TypeError("Missing reference particle orientation or position")
        positions = np.concatenate((ref_position,part_positions), axis=0)
        orientations = np.concatenate((ref_orientation,part_orientations), axis=0)
        particle_groups = [(np.array([p]),np.array([o])) for p,o in zip(positions,orientations)]
        return particle_groups

    def get_ref_orientation(self):
        """
        Return the orientation.
        """
        if self._ref_orientation is None:
            return psg.get_random_quat(1)

        ref_orientation = self._ref_orientation

        return ref_orientation

    def set_ref_orientation(self, ref_orientation):
        self._ref_orientation = ref_orientation

    def get_ref_position(self):
        """
        Return the position.
        """
        if self._ref_position is None:
            return random_positions_in_beam(1, 
                                            min(self.jet_radius, self.ref_jet_radius),
                                            max(self.jet_radius, self.ref_jet_radius))

        ref_position = self._ref_position

        return ref_position

    def set_ref_position(self, ref_position):
        self._ref_position = ref_position

    def get_next_part_orientation(self):
        """
        Return the next orientation.
        """
        if self._part_orientations is None:
            return psg.get_random_quat(self.n_target_particles)

        if self._i_part_orientations >= len(self._part_orientations):
            raise StopIteration("No more orientation available.")

        part_orientation = self._part_orientations[self._i_part_orientations:
                                                   self._i_part_orientations+self.n_target_particles]

        self._i_part_orientations += self.n_target_particles
        return part_orientation

    def set_part_orientations(self, part_orientations):
        self._part_orientations = part_orientations
        self._i_part_orientations = 0

    def get_next_part_position(self):
        """
        Return the next position.
        """
        if self._part_positions is None:
            particle_dict = dict(Counter(self.particles))
            states, coords = distribute_particles(particle_dict, 
                                                  min(self.jet_radius, self.ref_jet_radius), 
                                                  max(self.jet_radius, self.ref_jet_radius), 
                                                  sticking=False, iteration=0, max_iter=10)
            return coords

        if self._i_part_positions >= len(self._part_positions):
            raise StopIteration("No more position available.")

        part_position = self._part_positions[self._i_part_positions:
                                             self._i_part_positions+self.n_target_particles]

        self._i_part_positions += self.n_target_particles
        return part_position

    def set_part_positions(self, part_positions):
        self._part_positions = part_positions
        self._i_part_positions = 0
