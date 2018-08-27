import numpy as np
import pysingfel.geometry as pg
import pysingfel.util as pu
import pysingfel.diffraction as pd
import pysingfel.crossTalk as pc
import pysingfel.gpu.diffraction as pgd

import sys
import os

from PSCalib.GenericCalibPars import GenericCalibPars
from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays

"""
The idea behind these classes is that I can not put everything in a single class in a controllable 
and elegant way. Therefore, I try to create several different classes sharing the same interface
for the user and the other modules.

In these classes, in the __init__ function, properties and methods starting with _ are private which
means the other classes does not depends on them. 
"""


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
        self.pixel_num_x = 0  # number of pixels in x
        self.pixel_num_y = 0  # number of pixels in y
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
        self.pixel_index_x_max = 1
        self.pixel_index_y_max = 1

        # Corrections
        self.solid_angle_per_pixel = None  # solid angle
        self.polarization_correction = None  # Polarization correction

        """
        The theoretical differential cross section of an electron ignoring the polarization effect is,
                do/dO = ( e^2/(4*Pi*epsilon0*m*c^2) )^2  *  ( 1 + cos(xi)^2 )/2 
        Therefore, one needs to includes the leading constant factor which is the following numerical value.
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
        intensity = beam.get_photonsPerPulse() / (4 * beam.get_focus() ** 2)

        # Get the reciprocal positions and the corrections
        (self.pixel_position_reciprocal,
         self.pixel_distance_reciprocal,
         self.polarization_correction,
         self.solid_angle_per_pixel) = pg.reciprocal_position_and_correction(pixel_center=self.pixel_position,
                                                                             polarization=polar,
                                                                             wave_vector=wavevector,
                                                                             pixel_width=self.pixel_width,
                                                                             pixel_height=self.pixel_height,
                                                                             orientation=self.orientation)

        # Put all the corrections together
        self.linear_correction = intensity * self.Thomson_factor * np.multiply(self.polarization_correction,
                                                                               self.solid_angle_per_pixel)

    ####################################################################################################################
    # Calculate diffraction patterns
    ####################################################################################################################

    def get_pattern_without_corrections(self, particle, device="cpu"):
        """
        Generate a single diffraction pattern without any correction from the particle object.

        :param particle: The particle object
        :param device: 'cpu' or 'gpu'
        :return: A diffraction pattern.
        """

        if device == "cpu":
            diffraction_pattern = pd.calculate_molecularFormFactorSq(particle,
                                                                     self.pixel_distance_reciprocal,
                                                                     self.pixel_position_reciprocal)
        elif device == "gpu":
            diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(self.pixel_position_reciprocal,
                                                                        particle,
                                                                        "intensity")
        else:
            print(" The device parameter can only be set as \"gpu\" or \"cpu\" ")
            raise Exception('Wrong parameter value. device can only be set as \"gpu\" or \"cpu\" ')

        return diffraction_pattern

    def get_diffraction_field(self, particle, device="cpu"):
        """
        Generate a single diffraction pattern without any correction from the particle object.

        :param particle: The particle object
        :param device: 'cpu' or 'gpu'
        :return: A diffraction pattern.
        """

        if device == "cpu":
            diffraction_pattern = pd.calculate_molecularFormFactorSq(particle,
                                                                     self.pixel_distance_reciprocal,
                                                                     self.pixel_position_reciprocal)
        elif device == "gpu":
            diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(self.pixel_position_reciprocal,
                                                                        particle,
                                                                        "intensity")
        else:
            print(" The device parameter can only be set as \"gpu\" or \"cpu\" ")
            raise Exception('Wrong parameter value. device can only be set as \"gpu\" or \"cpu\" ')

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

    def get_photons(self, particle, device="cpu"):
        """
        Get a simulated photon patterns stack
        :param particle: The paticle object
        :param device: 'cpu' or 'gpu'
        :return: A image stack of photons
        """
        raw_data = self.get_pattern_without_corrections(particle=particle, device=device)
        return self.add_correction_and_quantization(raw_data)

    def get_adu(self, particle, path, device="cpu"):
        """
        Get a simulated adu pattern stack

        :param particle: The particle object.
        :param path: The path to the crosstalk effect library.
        :param device: 'cpu' or 'gpu'
        :return: An image stack of adu.
        """
        raw_photon = self.get_photons(particle=particle, device=device)
        return pc.add_cross_talk_effect_panel(lib_path=path, photons=raw_photon)

    ####################################################################################################################
    # For 3D slicing.
    ####################################################################################################################
    def preferred_voxel_length(self, wave_vector):
        """
        If one want to put the diffraction pattern into 3D reciprocal space, then one needs to select a
        proper voxel length for the reciprocal space. This function gives a reasonable estimation of this
        length

        :param wave_vector: The wavevector of in this experiment.
        :return: voxel_length.
        """
        # Notice that this voxel length has nothing to do with the voxel length utilized in dragonfly.
        voxel_length = np.sqrt(np.sum(np.square(wave_vector)))
        voxel_length /= self.distance * np.min(self.pixel_width, self.pixel_height)

        return voxel_length

    def preferred_reciprocal_mesh_number(self, wave_vector):
        """
        If one want to put the diffraction pattern into 3D reciprocal space, then one needs to select a
        proper voxel number for a proper voxel length for the reciprocal space.
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
        Get the a proper reciprocal mesh.

        :param voxel_number_1d: The voxel number along 1 dimension. Notice that this number has to be odd.
        :return: The reciprocal mesh, voxel length.
        """
        voxel_half_number = int((voxel_number_1d / 2) - 1)
        voxel_length = np.max(self.pixel_distance_reciprocal) / voxel_half_number
        voxel_number = int(voxel_number_1d) + 4  # Get some more pixels for flexibility

        return pg.get_reciprocal_mesh(voxel_number, voxel_length), voxel_length


class PlainDetector(DetectorBase):
    """
    This object constructs a detector based on the .geom file.
    """

    def __init__(self, geom_file=None, beam=None):
        """
        Define parameters.
        :param geom_file: The geometry file that can be used to initialize the object.
        :param beam: The beam object.
        """
        super(PlainDetector, self).__init__()
        if geom_file:
            if beam:
                print("Initialize the detector with the geometry file and the beam object.")
            else:
                print("Initialize the detector with the geometry file only. "
                      "Please also initialize the detector with the beam object."
                      " file before calculating the diffraction patterns. ")
            self.initialize(geom_file=geom_file, beam=beam)
        else:
            print("Please initialize the detector with the geometry file and the beam object with"
                  "self.initialize (and perhaps self.initialize_pixels_with_beam) ")

    def initialize(self, geom_file, beam=None):
        """
        Initialize the detector with the user-defined geometry file (and perhaps self.initialize_pixels_with_beam).

        :param geom_file: The path of the .geom file.
        :param beam: The beam object.
        :return: None
        """
        geom = pu.readGeomFile(geom_file)
        self.geometry = geom

        # Set parameters
        self.panel_num = 1

        # Extract info
        self.pixel_num_x = int(geom['pixel number x'])
        self.pixel_num_y = int(geom['pixel number y'])
        self.pixel_num_total = np.array([self.pixel_num_x * self.pixel_num_y, ])
        self.distance = np.array([geom['distance'], ])

        self.pixel_width = np.ones((self.panel_num, self.pixel_num_x, self.pixel_num_y)) * geom['pixel size x']
        self.pixel_height = np.ones((self.panel_num, self.pixel_num_x, self.pixel_num_y)) * geom['pixel size y']

        # Calculate real space position
        self.pixel_position = np.zeros((self.panel_num, self.pixel_num_x, self.pixel_num_y, 3))
        # z direction position
        self.pixel_position[0, ::, ::, 2] += self.distance

        # x,y direction position
        total_length_x = (self.pixel_num_x - 1) * self.pixel_width
        total_length_y = (self.pixel_num_y - 1) * self.pixel_height

        x_coordinate_temp = np.linspace(-total_length_x / 2, total_length_x / 2, num=self.pixel_num_x, endpoint=True)
        y_coordinate_temp = np.linspace(-total_length_y / 2, total_length_y / 2, num=self.pixel_num_y, endpoint=True)
        mesh_temp = np.meshgrid(x_coordinate_temp, y_coordinate_temp)

        self.pixel_position[0, ::, ::, 0] = mesh_temp[0][::, ::]
        self.pixel_position[0, ::, ::, 1] = mesh_temp[1][::, ::]

        # Calculate the index map for the image
        mesh_temp = np.meshgrid(np.arange(self.pixel_num_x), np.arange(self.pixel_num_y))
        self.pixel_index_map[0, :, :, 0] = mesh_temp[0][::, ::]
        self.pixel_index_map[0, :, :, 1] = mesh_temp[1][::, ::]
        self.pixel_index_x_max = self.pixel_num_x
        self.pixel_index_y_max = self.pixel_num_y

        # Initialize the detector effect parameters
        self.pedestal = np.zeros((1, self.pixel_num_x, self.pixel_num_y))
        self.pixel_rms = np.zeros((1, self.pixel_num_x, self.pixel_num_y))
        self.pixel_bkgd = np.zeros((1, self.pixel_num_x, self.pixel_num_y))
        self.pixel_status = np.zeros((1, self.pixel_num_x, self.pixel_num_y))
        self.pixel_mask = np.zeros((1, self.pixel_num_x, self.pixel_num_y))
        self.pixel_gain = np.ones((1, self.pixel_num_x, self.pixel_num_y))

        if beam:
            self.initialize_pixels_with_beam(beam=beam)

    def assemble_image_stack(self, image_stack):
        """
        Assemble the image stack into a 2D diffraction pattern.
        For this specific object, since it only has one panel, the result is to remove the first dimension.

        :param image_stack: The [1, num_x, num_y] numpy array.
        :return: The [num_x, num_y] numpy array.
        """
        return np.reshape(image_stack, (self.pixel_num_x, self.pixel_num_y))

    def assemble_image_stack_batch(self, image_stack_batch):
        """
        Assemble the image stack batch into a stack of 2D diffraction patterns.
        For this specific object, since it has only one panel, the result is a simple reshape.

        :param image_stack_batch: The [stack_num, 1, num_x, num_y] numpy array
        :return: The [stack_num, num_x, num_y] numpy array
        """
        stack_num = image_stack_batch.shape[0]
        return np.reshape(image_stack_batch, (stack_num, self.pixel_num_x, self.pixel_num_y))


class LclsDetector(DetectorBase):
    """
    Class for lcls detectors.
    """

    def __init__(self, geom=None, beam=None):
        """
        Initialize the detector.
        """
        super(LclsDetector, self).__init__()

        if geom:
            if beam:
                print("Initialize the detector with the geometry file and the beam object.")
            else:
                print("Initialize the detector with the geometry file only. "
                      "Please also initialize the detector with the beam object."
                      " file before calculating the diffraction patterns. ")
            self.initialize_as_lcls_detector(path=geom, beam=beam)
        else:
            print("Please initialize the detector with the geometry file and the beam object with"
                  "self.initialize (and perhaps self.initialize_pixels_with_beam) ")

    def initialize_as_lcls_detector(self, path, beam):

        # parse the path to extract the necessary information to use psana modules
        parsed_path = path.split('/')

        # notify the user that the path should be as deep as the geometry folder
        if parsed_path[-1] != "geometry":
            # print parsed_path[-1]
            print(" Sorry, at present, the package is not very smart. Please specify the path of the detector" +
                  "as deep as the geometry folder. And example would be like:" +
                  "/reg/d/psdm/amo/experiment_name/calib/group/source/geometry" +
                  "where the \" /calib/group/source/geometry \" part is essential. The address before that part is not"
                  + " essential and can be replaced with your absolute address or relative address.")

        fname_geometry = path + '/0-end.data'

        # At present I have to redirect the output to make the output reasonable to user
        old_stdout = sys.stdout
        f = open('Detector_initialization.log', 'w')
        sys.stdout = f

        self.geometry = GeometryAccess(fname_geometry, 0o377)

        #################################################################
        # The default geometry information may require some modification
        #################################################################

        # All pre-defined detector are initialized in a similar way.
        # We will parse the path in the self._initialize(path) function again
        self.initialize_as_pnccd(path)

        sys.stdout = old_stdout
        f.close()
        os.remove('./Detector_initialization.log')

        if beam:
            self.initialize_pixels_with_beam(beam=beam)

    def initialize_as_pnccd(self, path):

        #################################################################
        # The following several lines initialize the geometry information
        #################################################################
        # Set coordinate in real space
        temp = self.geometry.get_pixel_coords()
        temp_index = self.geometry.get_pixel_coord_indexes()

        self.panel_num = temp[0].shape[1] * temp[0].shape[2]
        self.distance = temp[2][0, 0, 0, 0, 0]

        self.pixel_position = np.zeros((self.panel_num, temp[0].shape[3], temp[0].shape[4], 3))
        self.pixel_index_map = np.zeros((self.panel_num, temp[0].shape[3], temp[0].shape[4], 2))

        for l in range(temp[0].shape[1]):
            for m in range(temp[0].shape[2]):
                for n in range(3):
                    self.pixel_position[m + l * temp[0].shape[2], :, :, n] = temp[n][0, l, m, :, :]
                for n in range(2):
                    self.pixel_index_map[m + l * temp[0].shape[2], :, :, n] = temp_index[n][0, l, m, :, :]

        self.pixel_index_map = self.pixel_index_map.astype('int32')

        del temp
        del temp_index

        self.panel_orientation = np.array([[0, 0, 1], ] * self.panel_num)
        self.pixel_num_x = np.array([self.pixel_index_map.shape[1], ] * self.panel_num)
        self.pixel_num_y = np.array([self.pixel_index_map.shape[2], ] * self.panel_num)
        self.pix_num_total = np.sum(np.multiply(self.pixel_num_x, self.pixel_num_y))

        tmp = float(self.geometry.get_pixel_scale_size())
        self.pixel_width = np.ones((self.panel_num, self.pixel_num_x[0], self.pixel_num_y[0])) * tmp
        self.pixel_height = np.ones((self.panel_num, self.pixel_num_x[0], self.pixel_num_y[0])) * tmp

        ##################################
        # The following several lines initialize the detector effects besides cross talk.
        ##################################
        # first we should parse the path
        parsed_path = path.split('/')

        cbase = CalibParsBasePnccdV1()
        calibdir = '/'.join(parsed_path[:-3])
        group = parsed_path[-3]
        source = parsed_path[-2]
        runnum = 0
        pbits = 255
        gcp = GenericCalibPars(cbase, calibdir, group, source, runnum, pbits)

        self.pedestal = gcp.pedestals()
        self.pixel_rms = gcp.pixel_rms()
        self.pixel_mask = gcp.pixel_mask()
        self.pixel_bkgd = gcp.pixel_bkgd()
        self.pixel_status = gcp.pixel_status()
        self.pixel_gain = gcp.pixel_gain()

    def assemble_image_stack(self, image_stack):
        # construct the image holder:
        image = np.zeros((self.pixel_index_x_max, self.pixel_index_y_max))
        for l in range(self.panel_num):
            image[self.pixel_index_map[l, :, :, 0],
                  self.pixel_index_map[l, :, :, 1]] = image_stack[l, :, :]

        return image


class UserDefinedDetector(DetectorBase):
    """
    Class for lcls detectors.
    """

    def __init__(self):
        """
        Initialize the detector.
        """
        super(UserDefinedDetector, self).__init__()

    def initialize(self, param):
        """
        Initialize the detector with user defined parameters
        :param param: The dictionary containing all the necessary information to initialized the detector.
        :return: None
        """

        """
        Doc:
            To use this class, the user has to provide the necessary information to initialize the detector.
            All the necessary entries are listed in the example notebook.
        """
        pass
