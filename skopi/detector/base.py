import numpy as np
import os
import sys

import skopi as ps
import skopi.geometry as pg
import skopi.util as pu
from skopi.util import deprecation_message, xp
from skopi import particle
from skopi.particlePlacement import * #max_radius, distribute_particles
from skopi.geometry import * #quaternion2rot3d, get_random_rotation, get_random_translations


class DetectorBase(object):
    """
    This is the base object for all detector object.
    This class contains some basic operations of the detector.
    It provides interfaces for the other modules.
    """

    def __init__(self):
        # Reciprocal space information is only available if detector has access to beam.
        self._has_beam = False

        # Define the hierarchy system. For simplicity, we only use two-layer structure.
        self.panel_num = 1

        # Define all properties the detector should have
        self._distance = 1  # (m) detector distance
        self.pixel_width = 0  # (m)
        self.pixel_height = 0  # (m)
        self.pixel_area = 0  # (m^2)
        self.panel_pixel_num_x = 0  # number of pixels in x
        self.panel_pixel_num_y = 0  # number of pixels in y
        self.pixel_num_total = 0  # total number of pixels (px*py)
        self.center_x = 0  # center of detector in x
        self.center_y = 0  # center of detector in y
        self.orientation = xp.array([0, 0, 1])
        self.pixel_position = None  # (m)
        self.pixel_position_ideal = None #(m)

        # pixel information in reciprocal space
        self.pixel_position_reciprocal = None  # (m^-1)
        self.pixel_distance_reciprocal = None  # (m^-1)

        # Pixel map
        self.pixel_index_map = None
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
        self._pedestal = 0
        self._pixel_rms = 0
        self._pixel_bkgd = 0
        self._pixel_status = 0
        self._pixel_mask = 0
        self._pixel_gain = 0

        # self.geometry currently only work for the pre-defined detectors
        self.geometry = None

    @property
    def distance(self):
        """Sample-detector distance."""
        return self._distance

    @distance.setter
    def distance(self, value):
        if not xp.allclose(self.orientation, xp.array([0, 0, 1])):
            raise NotImplementedError(
                "Detector distance setter only implemented for "
                "detector orientations along the z axis.")
        self.pixel_position[..., 2] *= value/self._distance
        self._distance = value
        if self._has_beam:  # Update pixel_position_reciprocal & co
            self.initialize_pixels_with_beam(beam=self._beam)

    @property
    def shape(self):
        """Unassembled detector shape."""
        return self._shape

    @property
    def pedestals(self):
        return self._pedestals

    @property
    def pixel_rms(self):
        return self._pixel_rms

    @property
    def pixel_mask(self):
        return self._pixel_mask

    @property
    def pixel_bkgd(self):
        return self._pixel_bkgd

    @property
    def pixel_status(self):
        return self._pixel_status

    @property
    def pixel_gain(self):
        return self._pixel_gain

    def initialize_pixels_with_beam(self, beam=None):
        """
        Calculate the pixel position in the reciprocal space and several corrections.
        :param beam: The beam object
        :return: None
        """
        if beam is None:
            return

        self._has_beam = True
        self._beam = beam

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
        self.linear_correction = intensity * self.Thomson_factor * xp.multiply(
            self.polarization_correction,
            self.solid_angle_per_pixel)

    def ensure_beam(self):
        if not self._has_beam:
            raise RuntimeError("This Detector hasn't been initialized with a beam.")

    ###############################################################################################
    # Calculate diffraction patterns
    ###############################################################################################

    def get_pattern_without_corrections(self, particle, device=None, return_type="intensity"):
        """
        Generate a single diffraction pattern without any correction from the particle object.

        :param particle: The particle object.
        :return: A diffraction pattern.
        """
        self.ensure_beam()

        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        import skopi.gpu.diffraction as pgd  # Only import GPU if needed
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
        self.ensure_beam()

        if device:
            deprecation_message(
                "Device option is deprecated. "
                "Everything now runs on the GPU.")

        import skopi.gpu.diffraction as pgd  # Only import GPU if needed
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
        self.ensure_beam()
        pattern = xp.asarray(pattern)
        displ = xp.asarray(displ)
        return pattern * xp.exp(1j * xp.dot(2*np.pi*self.pixel_position_reciprocal, displ))

    def add_static_noise(self, pattern):
        """
        Add static noise to the diffraction pattern.
        :param pattern: The pattern stack.
        :return: Pattern stack + static_noise
        """
        return pattern + xp.random.uniform(0, 2 * xp.sqrt(3 * self.pixel_rms))

    def reset_beam_center(self):
        """
        Reset pixel positions by returning beam to center of detector.
        """
        if self.pixel_position_ideal is not None:
            self.pixel_position = self.pixel_position_ideal
        return

    def offset_beam_center(self, sigma):
        """
        Add beam jitter to the pixel positions, assuming independent
        Gaussian diplacements along x and y from the beam center.
        :param sigma: standard deviation of Gaussian in pixels
        :return disp: (x,y) displacements in pixels from ideal center
        """
        self.ensure_beam()

        # track or reset to ideal pixel positions, without jitter
        if self.pixel_position_ideal is None:
            self.pixel_position_ideal = self.pixel_position.copy()
        else:
            self.pixel_position = self.pixel_position_ideal.copy()

        # offset pixel positions by reinitializing beam
        scale = np.mean(self.pixel_width)
        xdisp, ydisp = scale * np.random.normal(loc=0, scale=sigma, size=2)
        self.pixel_position += np.array([xdisp, ydisp, 0])
        self.initialize_pixels_with_beam(beam=self._beam)

        return (xdisp / scale, ydisp / scale)

    def add_solid_angle_correction(self, pattern):
        """
        Add solid angle corrections to the image stack.
        :param pattern: Pattern stack
        :return: Pattern stack with solid angle correction
        """
        self.ensure_beam()
        return xp.multiply(pattern, self.solid_angle_per_pixel)

    def add_polarization_correction(self, pattern):
        """
        Add polarization correction to the image stack
        :param pattern: image stack
        :return: image stack with polarization correction applied
        """
        self.ensure_beam()
        return xp.multiply(pattern, self.polarization_correction)

    def add_correction(self, pattern):
        """
        Add linear correction to the image stack
        :param pattern: The image stack
        :return: image stack with linear correction applied
        """
        self.ensure_beam()
        return xp.multiply(pattern, self.linear_correction)

    def add_quantization(self,pattern):
        """
        Apply quantization to the image stack
        :param pattern: The image stack
        :return: quantized image stack
        """
        return xp.random.poisson(pattern)

    def add_correction_and_quantization(self, pattern):
        """
        Add corrections to image stack and apply quantization to the image stack
        :param pattern: The image stack.
        :return: images with linear correction applied and quantized
        """
        self.ensure_beam()
        return xp.random.poisson(xp.multiply(pattern, self.linear_correction))

    def add_correction_batch(self,pattern_batch):
        """
        Add corrections to a batch of image stack
        :param pattern_batch [image stack index,image stack shape]
        :return:
        """
        self.ensure_beam()
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
        self.ensure_beam()
        return xp.random.poisson(xp.multiply(pattern_batch, self.linear_correction[xp.newaxis]))

    def remove_polarization(self, img, res=None):
        """
        Account for the effects of polarization.
        :param img: assembled (2D) or unassembled (3D) diffraction image
        :param res: diffraction resolution in Angstroms
        :return img_corr: image corrected for the effects of polarization to given resolution
        """
        self.ensure_beam()
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

    ###############################################################################################
    # For 3D slicing: computing full diffraction volume and slicing to compute pixels' intensities.
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
        self.ensure_beam()

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
        self.ensure_beam()
        dist_max = xp.max(self.pixel_distance_reciprocal)
        return pg.get_reciprocal_mesh(voxel_number_1d, dist_max)

    ###############################################################################################
    # For (dis)assembly of detector image from constituent panels and back.
    ###############################################################################################

    def assemble_image_stack(self, image_stack):
        """
        Assemble the image stack into a 2D diffraction pattern.
        For this specific object, since it only has one panel, the result is to remove the
        first dimension.

        :param image_stack: The [1, num_x, num_y] numpy array.
        :return: The [num_x, num_y] numpy array.
        """
        if self.pixel_index_map is None:
            raise RuntimeError(
                "This detector does not have pixel mapping information.")
        # construct the image holder:
        image = xp.zeros((self.detector_pixel_num_x, self.detector_pixel_num_y))
        for l in range(self.panel_num):
            image[self.pixel_index_map[l, :, :, 0],
                  self.pixel_index_map[l, :, :, 1]] = image_stack[l, :, :]

        return image

    def assemble_image_stack_batch(self, image_stack_batch):
        """
        Assemble the image stack batch into a stack of 2D diffraction patterns.
        For this specific object, since it has only one panel, the result is a simple reshape.

        :param image_stack_batch: The [stack_num, 1, num_x, num_y] numpy array
        :return: The [stack_num, num_x, num_y] numpy array
        """
        if self.pixel_index_map is None:
            raise RuntimeError(
                "This detector does not have pixel mapping information.")

        stack_num = image_stack_batch.shape[0]

        # construct the image holder:
        image = xp.zeros((stack_num, self.detector_pixel_num_x, self.detector_pixel_num_y))
        for l in range(self.panel_num):
            idx_map_1 = self.pixel_index_map[l, :, :, 0]
            idx_map_2 = self.pixel_index_map[l, :, :, 1]
            image[:, idx_map_1, idx_map_2] = image_stack_batch[:, l]

        return image

    def disassemble_image_stack(self, image):
        """
        Diassemble the 2D diffraction pattern into its consituent panels. For the base
        object with 1 panel, this merely reshapes the image by adding an axis.

        :param image: image of shape [detector_pixel_num_x, detector_pixel_num_y]
        :return image_stack: stack of shape [panel_num, panel_pixel_num_x, panel_pixel_num_y]
        """
        if self.pixel_index_map is None:
            raise RuntimeError("This detector does not have pixel mapping information.")
    
        image_stack = np.zeros((self.panel_num, self.pixel_index_map.shape[1], self.pixel_index_map.shape[2]))
    
        for panel in range(self.panel_num):
            idx_map_1 = self.pixel_index_map[panel, :, :, 0]
            idx_map_2 = self.pixel_index_map[panel, :, :, 1]
            image_stack[panel] = image[idx_map_1,idx_map_2]
        
        return image_stack

    def disassemble_image_stack_batch(self, image):
        """
        Diassemble a series of 2D diffraction patterns into their consituent panels. 

        :param image: images of shape [stack_num, detector_pixel_num_x, detector_pixel_num_y]
        :return image_stack_batch: array of shape [stack_num, panel_num, panel_pixel_num_x, panel_pixel_num_y]
        """
        if self.pixel_index_map is None:
            raise RuntimeError("This detector does not have pixel mapping information.")
 
        image_stack_batch = np.zeros((image.shape[0], self.panel_num, 
                                      self.pixel_index_map.shape[1], self.pixel_index_map.shape[2]))
    
        for panel in range(self.panel_num):
            idx_map_1 = self.pixel_index_map[panel, :, :, 0]
            idx_map_2 = self.pixel_index_map[panel, :, :, 1]
            image_stack_batch[:,panel] = image[:,idx_map_1,idx_map_2]
        
        return image_stack_batch
