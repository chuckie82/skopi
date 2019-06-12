import numpy as np
import os
import sys

from PSCalib.GenericCalibPars import GenericCalibPars
from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays

import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.crosstalk as pc
import pysingfel.gpu.diffraction as pgd
from pysingfel.util import deprecation_message


class DetectorBase(object):
    """
    This is the base object for all detector object.
    This class contains some basic operations of the detector.
    It provides interfaces for the other modules.
    """

    def __init__(self):
        # Define the hierarchy system. For simplicity, we only use two-layer structure.
        self.panel_num = 1

        # Define all properties the detector should have
        self.distance = 1  # (m) detector distance
        self.pixel_width = 0  # (m)
        self.pixel_height = 0  # (m)
        self.pixel_area = 0  # (m^2)
        self.panel_pixel_num_x = 0  # number of pixels in x
        self.panel_pixel_num_y = 0  # number of pixels in y
        self.pixel_num_total = 0  # total number of pixels (px*py)
        self.center_x = 0  # center of detector in x
        self.center_y = 0  # center of detector in y
        self.orientation = np.array([0, 0, 1])
        self.pixel_position = None  # (m)

        # pixel information in reciprocal space
        self.pixel_position_reciprocal = None  # (m^-1)
        self.pixel_distance_reciprocal = None  # (m^-1)

        # Pixel map
        self.pixel_index_map = 0
        self.detector_pixel_num_x = 1
        self.detector_pixel_num_y = 1

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

        # Detector effects
        self.pedestal = 0
        self.pixel_rms = 0
        self.pixel_bkgd = 0
        self.pixel_status = 0
        self.pixel_mask = 0
        self.pixel_gain = 0

        # self.geometry currently only work for the pre-defined detectors
        self.geometry = None

    def initialize_pixels_with_beam(self, beam=None):
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
            pixel_position=self.pixel_position,
            polarization=polar,
            wave_vector=wavevector,
            pixel_area=self.pixel_area,
            orientation=self.orientation)

        # Put all the corrections together
        self.linear_correction = intensity * self.Thomson_factor * np.multiply(
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

        diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(
            self.pixel_position_reciprocal,
            particle,
            "intensity")

        return np.multiply(diffraction_pattern, self.linear_correction)

    def add_static_noise(self, pattern):
        """
        Add static noise to the diffraction pattern.
        :param pattern: The pattern stack.
        :return: Pattern stack + static_noise
        """
        return pattern + np.random.uniform(0, 2 * np.sqrt(3 * self.pixel_rms))

    def add_solid_angle_correction(self, pattern):
        """
        Add solid angle corrections to the image stack.
        :param pattern: Pattern stack
        :return:
        """
        return np.multiply(pattern, self.solid_angle_per_pixel)

    def add_polarization_correction(self, pattern):
        """
        Add polarization correction to the image stack
        :param pattern: image stack
        :return:
        """
        return np.multiply(pattern, self.polarization_correction)

    def add_correction_and_quantization(self, pattern):
        """
        Add corrections to image stack and apply quantization to the image stack
        :param pattern: The image stack.
        :return:
        """
        return np.random.poisson(np.multiply(pattern, self.linear_correction))

    def add_correction_and_quantization_batch(self, pattern_batch ):
        """
        Add corrections to a batch of image stack and apply quantization to the batch

        :param pattern_batch: [image stack index, image stack shape]
        :return:
        """
        return np.random.poisson(np.multiply(pattern_batch, self.linear_correction[np.newaxis]))

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

        raw_data = self.get_pattern_without_corrections(particle=particle)
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
        voxel_length = np.sqrt(np.sum(np.square(wave_vector)))
        voxel_length /= self.distance * np.min(self.pixel_width, self.pixel_height)

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
        reciprocal_space_range = np.max(self.pixel_distance_reciprocal)
        # The voxel number along 1 dimension is 2*voxel_half_num_1d+1
        voxel_half_num_1d = int(np.floor_divide(reciprocal_space_range, voxel_length) + 1)

        voxel_num_1d = int(2 * voxel_half_num_1d + 1)
        return voxel_num_1d

    def get_reciprocal_mesh(self, voxel_number_1d):
        """
        Get the proper reciprocal mesh.

        :param voxel_number_1d: The voxel number along 1 dimension.
        :return: The reciprocal mesh, voxel length.
        """
        dist_max = np.max(self.pixel_distance_reciprocal)
        return pg.get_reciprocal_mesh(voxel_number_1d, dist_max)
