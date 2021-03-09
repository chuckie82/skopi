import numpy as np
import skopi.geometry as psg
from skopi.particlePlacement import *
from skopi.aggregate import build_bpca

from .base import Experiment


class SPIExperiment(Experiment):
    """
    Class for SPI experiment.
    """

    def __init__(self, det, beam, particle, n_part_per_shot=1, jet_radius=None):
        """
        Initialize a SPI experiment. Here we assume n_part_per_shot (default=1) of a single
        particle type. If more than one particle, then the particles form an aggregate that
        is randomly positioned in the volume formed by the intersection of the jet and beam.
        If an aggregate of different particle types is desired, the FXS class should be used
        instead with the sticking flag set to True.
        
        :param det: The detector object.
        :param beam: The beam object.
        :param particle: The particle object.
        :param jet_radius: The radius of the gas jet in meters.
        :param n_part_per_shot: The number of particles per shot, default is 1.
        """
        super(SPIExperiment, self).__init__(det, beam, [particle]*n_part_per_shot)
        self.jet_radius = jet_radius
        self.n_part_per_shot = n_part_per_shot
        self.particle_radius = max_radius([particle])
        self._orientations, self._positions = None, None

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"
        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        In the SPI case, it is n_part_per_shot of one particle type.
        """
        orientations = self.get_next_orientation()
        positions = self.get_next_position()
        particle_groups = [(np.array([p]),np.array([o])) for p,o in zip(positions,orientations)]
        return particle_groups

    def get_next_orientation(self):
        """
        Return the next orientation. If orientations were not explicitly
        set, then random orientations will be generated.

        :return orientation: array of n_part_per_shot quaternions
        """
        if self._orientations is None:
            return psg.get_random_quat(self.n_part_per_shot)

        if self._i_orientations >= len(self._orientations):
            raise StopIteration("No more orientation available.")
        
        orientation = self._orientations[self._i_orientations:
                                         self._i_orientations+self.n_part_per_shot]

        self._i_orientations += self.n_part_per_shot
        return orientation

    def set_orientations(self, orientations):
        """
        Set all orientations to be used by this instance of class.
        
        :param orientations: array of quaternions
        """
        self._orientations = orientations
        self._i_orientations = 0

    def get_next_position(self):
        """
        Return the next position.

        :return position: array of n_part_per_shot coordinates
        """
        if self._positions is None:
            center, offset = np.zeros((self.n_part_per_shot,3)), np.zeros((1,3))
            if self.n_part_per_shot > 1:
                aggregate = build_bpca(num_pcles=self.n_part_per_shot, radius=self.particle_radius)
                center = aggregate.pos
            if self.jet_radius is not None:
                offset = random_positions_in_beam(1, self.beam.get_focus()[0]/2, self.jet_radius)
            return center + offset

        if self._i_positions >= len(self._positions):
            raise StopIteration("No more position available.")

        position = self._positions[self._i_positions:
                                   self._i_positions+self.n_part_per_shot]

        self._i_positions += self.n_part_per_shot
        return position

    def set_positions(self, positions):
        """
        Set all particle positions to be used by this instance of the class.
        
        :param positions: array of coordinates in meters
        """
        self._positions = positions
        self._i_positions = 0

