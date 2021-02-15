import numpy as np
import os
import sys

import skopi as ps
import skopi.geometry as pg
import skopi.util as pu
import skopi.crosstalk as pc
from skopi.util import deprecation_message, xp
from skopi import particle
from skopi.particlePlacement import * #max_radius, distribute_particles
from skopi.geometry import * #quaternion2rot3d, get_random_rotation, get_random_translations


class ReciprocalDetector(object):
    """
    Base object for the detector in reciprocal space.

    Focus on wavelength-specific elements.
    """

    def __init__(self, detector, beam):
        self.detector = detector
        self.beam = beam

        # pixel information in reciprocal space
        self.pixel_position_reciprocal = None  # (m^-1)
        self.pixel_distance_reciprocal = None  # (m^-1)

        # Corrections
        self.solid_angle_per_pixel = None  # solid angle
        self.polarization_correction = None  # Polarization correction

        """
        The theoretical differential cross section of an electron ignoring the
        polarization effect is,
                do/dO = ( e^2/(4*Pi*epsilon0*m*c^2) )^2  *  ( 1 + cos(xi)^2 )/2
        Therefore, one needs to includes the leading constant factor which is the
        following numerical value.
        """
        # Tompson Scattering factor
        self.Thomson_factor = 2.817895019671143 * 2.817895019671143 * 1e-30

        # Total scaling and correction factor.
        self.linear_correction = None

        self.initialize_pixels_with_beam(beam)

    def initialize_pixels_with_beam(self, beam):
        """
        Calculate the pixel position in the reciprocal space and several corrections.
        :param beam: The beam object
        :return: None
        """
        wavevector = beam.get_wavevector()
        polar = beam.Polarization
        intensity = beam.get_photons_per_pulse() / beam.get_focus_area()

        # Get the reciprocal positions and the corrections
        (self.pixel_position_reciprocal,
         self.pixel_distance_reciprocal,
         self.polarization_correction,
         self.solid_angle_per_pixel) = pg.get_reciprocal_position_and_correction(
            pixel_position=self.detector.pixel_position,
            polarization=polar,
            wave_vector=wavevector,
            pixel_area=self.detector.pixel_area,
            orientation=self.detector.orientation)

        # Put all the corrections together
        self.linear_correction = intensity * self.Thomson_factor * xp.multiply(
            self.polarization_correction,
            self.solid_angle_per_pixel)

    ###############################################################################################
    # Calculate diffraction patterns
    ###############################################################################################

    def get_pattern_without_corrections(self, particle, device=None, return_type="intensity"):
        """
        Generate a single diffraction pattern without any correction from the particle object.

        :param particle: The particle object.
        :return: A diffraction pattern.
        """
        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        import skopi.gpu.diffraction as pgd
        diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(
            self.pixel_position_reciprocal,
            particle,
            return_type)

        return diffraction_pattern

    def get_intensity_field(self, particle, device=None):
        """
        Generate a single diffraction pattern without any correction from the particle object.

        :param particle: The particle object.
        :return: A diffraction pattern.
        """
        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        import skopi.gpu.diffraction as pgd
        diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(
            self.pixel_position_reciprocal,
            particle,
            "intensity")

        return xp.multiply(diffraction_pattern, self.linear_correction)

    def add_phase_shift(self, pattern, displ):
        """
        Add phase shift corresponding to displ to complex pattern.

        :param pattern: complex field pattern.
        :param displ: displ(acement) (position) of the particle (m).
        :return: modified complex field pattern.
        """
        pattern = xp.asarray(pattern)
        displ = xp.asarray(displ)
        return pattern * xp.exp(1j * xp.dot(self.pixel_position_reciprocal, displ))

    def add_solid_angle_correction(self, pattern):
        """
        Add solid angle corrections to the image stack.
        :param pattern: Pattern stack
        :return:
        """
        return xp.multiply(pattern, self.solid_angle_per_pixel)

    def add_polarization_correction(self, pattern):
        """
        Add polarization correction to the image stack
        :param pattern: image stack
        :return:
        """
        return xp.multiply(pattern, self.polarization_correction)

    def add_correction(self, pattern):
        """
        Add linear correction to the image stack
        :param pattern: The image stack
        :return:
        """
        return xp.multiply(pattern, self.linear_correction)

    def add_quantization(self,pattern):
        """
        Apply quantization to the image stack
        :param pattern: The image stack
        :return:
        """
        return xp.random.poisson(pattern)

    def add_correction_and_quantization(self, pattern):
        """
        Add corrections to image stack and apply quantization to the image stack
        :param pattern: The image stack.
        :return:
        """
        return xp.random.poisson(xp.multiply(pattern, self.linear_correction))

    def add_correction_batch(self,pattern_batch):
        """
        Add corrections to a batch of image stack
        :param pattern_batch [image stack index,image stack shape]
        :return:
        """
        return xp.multiply(pattern_batch, self.linear_correction[xp.newaxis])

    def add_quantization_batch(self,pattern_batch):
        """
        Add quantization to a batch of image stack
        :param pattern_batch [image stack index, image stack shape]
        :return:
        """
        return xp.random.poisson(pattern_batch)

    def add_correction_and_quantization_batch(self, pattern_batch):
        """
        Add corrections to a batch of image stack and apply quantization to the batch
        :param pattern_batch: [image stack index, image stack shape]
        :return:
        """
        return xp.random.poisson(xp.multiply(pattern_batch, self.linear_correction[xp.newaxis]))

    def remove_polarization(self, img, res=None):
        """
        img: assembled (2D) or unassembled (3D) diffraction image
        res: diffraction resolution in Angstroms
        """
        if res is not None:
            mask = self.pixel_distance_reciprocal < (1./res)
        else:
            mask = 1
        correction = mask * self.polarization_correction + (1-mask)
        return img / correction

    def get_photons(self, particle, device=None):
        """
        Get a simulated photon patterns stack
        :param particle: The paticle object
        :param device: 'cpu' or 'gpu'
        :return: A image stack of photons
        """
        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        raw_data = self.get_pattern_without_corrections(particle=particle,return_type="intensity")
        return self.add_correction_and_quantization(raw_data)

    def get_adu(self, particle, path, device=None):
        """
        Get a simulated adu pattern stack

        :param particle: The particle object.
        :param path: The path to the crosstalk effect library.
        :param device: 'cpu' or 'gpu'
        :return: An image stack of adu.
        """
        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        raw_photon = self.get_photons(particle=particle)
        return pc.add_cross_talk_effect_panel(db_path=path, photons=raw_photon)

    ###############################################################################################
    # For 3D slicing.
    ###############################################################################################

    def preferred_voxel_length(self, wave_vector):
        """
        If one want to put the diffraction pattern into 3D reciprocal space, then one needs to
        select a proper voxel length for the reciprocal space. This function gives a reasonable
        estimation of this length

        :param wave_vector: The wavevector of in this experiment.
        :return: voxel_length.
        """
        # Notice that this voxel length has nothing to do with the voxel length
        # utilized in dragonfly.
        voxel_length = xp.sqrt(xp.sum(xp.square(wave_vector)))
        voxel_length /= self.distance * xp.min(self.pixel_width, self.pixel_height)

        return voxel_length


    def preferred_reciprocal_mesh_number(self, wave_vector):
        """
        If one want to put the diffraction pattern into 3D reciprocal space, then one needs to
        select a proper voxel number for a proper voxel length for the reciprocal space.
        This function gives a reasonable estimation of this length and voxel number

        :param wave_vector: The wavevector of in this experiment.
        :return: The reciprocal mesh number along 1 dimension
        """
        """ Return the prefered the reciprocal voxel grid number along 1 dimension. """
        voxel_length = self.preferred_voxel_length(wave_vector)
        reciprocal_space_range = xp.max(self.pixel_distance_reciprocal)
        # The voxel number along 1 dimension is 2*voxel_half_num_1d+1
        voxel_half_num_1d = int(xp.floor_divide(reciprocal_space_range, voxel_length) + 1)

        voxel_num_1d = int(2 * voxel_half_num_1d + 1)
        return voxel_num_1d


    def get_reciprocal_mesh(self, voxel_number_1d):
        """
        Get the proper reciprocal mesh.

        :param voxel_number_1d: The voxel number along 1 dimension.
        :return: The reciprocal mesh, voxel length.
        """
        dist_max = xp.max(self.pixel_distance_reciprocal)
        return pg.get_reciprocal_mesh(voxel_number_1d, dist_max)

