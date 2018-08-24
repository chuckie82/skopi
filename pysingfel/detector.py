import psana
import numpy as np
import pysingfel.geometry
import pysingfel.util
import pysingfel.diffraction as pd

from pysingfel.crossTalk import add_cross_talk_effect_panel as cross_talk_panel

import sys
import os

# USE :py:class:`PSCalib.CalibParsStore`
import PSCalib.GlobalUtils as gu
from PSCalib.GenericCalibPars import GenericCalibPars
from PSCalib.CalibParsBaseAndorV1 import CalibParsBaseAndorV1
from PSCalib.CalibParsBaseCameraV1 import CalibParsBaseCameraV1
from PSCalib.CalibParsBaseCSPad2x2V1 import CalibParsBaseCSPad2x2V1
from PSCalib.CalibParsBasePnccdV1 import CalibParsBasePnccdV1
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays


# The information the detector module should contain
class Detector(object):
    def __init__(self, mode="Unspecified"):

        # whether use the user defined detector or pre-defined detector
        self.mode = mode

        # Flags
        self.detector_initialized = 0
        self.pixel_initialized = 0

        # Define the hierarchy system. For simplicity, we only use two-layer structure.
        self.panel_num = 1
        self.panel_orientation = None

        # Define all properties the detector should have
        self.distance = 0  # (m) detector distance
        self.pix_width = 0  # (m)
        self.pix_height = 0  # (m)
        self.pix_area = 0  # (m^2)
        self.pix_num_x = 0  # number of pixels in x
        self.pix_num_y = 0  # number of pixels in y
        self.pix_num_total = 0  # total number of pixels (px*py)
        self.center_x = 0  # center of detector in x
        self.center_y = 0  # center of detector in y
        self.orientation = [0, 0, 1]

        # pixel position in real space
        self.pixel_position = None

        # pixel information in reciprocal space
        self.pixel_position_reciprocal = None  # (m^-1)
        self.pixel_distance_reciprocal = None  # pixel distance to the center in reciprocal space
        self.pixel_index_map = 0

        # Corrections
        self.solid_angle_correction = None  # solid angle
        self.polarization_correction = None  # Polarization correction 

        # Detector effects
        self.pedestal = 0
        self.pixel_rms = 0
        self.pixel_bkgd = 0
        self.pixel_status = 0
        self.pixel_mask = 0
        self.pixel_gain = 0

        # self.geometry currently only work for the pre-defined detectors
        self.geometry = None

        if not ((self.mode == "User Defined") or (self.mode == "LCLS Detectors")):
            print('If you want to use LCLS detectors like pnccdFront or pnccdBack,\n ' +
                  'please specifiy the mode parameter as \"LCLS Detector\".' +
                  'After the creation of the detector object, please use ' +
                  'the self.initialize_as_LCLS_detector(path= path of the detector calib folder) function' +
                  'to initialize the detector object.')
            print('If you don\'t want to use LCLS detector and only want to set up a \n' +
                  'plain panel of detector with specified pixel size and distance to the interaction \n' +
                  'point, please set the mode parameter as \"User Defined\". Then please use the function: \n' +
                  ' self.initialize_with_detector_geometry_file(path= path of the detector geometry file)')
            print('By the way, advanced user can tune the .data file associated with the LCLS detector to change \n' +
                  'detector parameters. The detailed methods can be found \n' +
                  'in the official documentation.')

        if self.mode == "User Defined":
            print("Please use self.initialize_with_detector_geometry_" +
                  "file(path= path of the detector geometry file) to " +
                  "initialize the detector object.")

        if self.mode == "LCLS Detectors":
            print(
                "Please use self.initialize_as_LCLS_" +
                "detector(path= path of the detector calib folder)" +
                " to initialize the detector object.")

    def initialize_with_detector_geometry_file(self, path):

        self.mode = "User Defined"
        # Change the flag self.detector_initialized
        self.detector_initialized = 1

        geom = pysingfel.util.readGeomFile(path)
        # Set paramters
        self.panel_num = 1
        # Notice that currently the position is calculated with the assumption that the orientation is [0,0,1]
        # So the following value should not be changed at present.
        self.panel_orientation = np.array([[0, 0, 1], ])

        self.pix_num_x = int(geom['pixel number'])
        self.pix_num_y = int(geom['pixel number'])
        self.pix_num_total = np.array([self.pix_num_x * self.pix_num_y, ])

        self.distance = np.array([geom['distance'], ])

        self.pix_width = np.ones((1, self.pix_num_x, self.pix_num_y)) * geom['pixel size']
        self.pix_height = np.ones((1, self.pix_num_x, self.pix_num_y)) * geom['pixel size']

        # Calculate real space position
        self.pixel_position = np.zeros((1, self.pix_num_x, self.pix_num_y, 3))
        # z direction position
        self.pixel_position[0, ::, ::, 2] += self.distance

        # These variables will be automatically released when the function finishes
        # So do not bother to delete them manually.
        x_coordinate_temp = self.pix_width * (np.array(range(self.pix_num_x)) - (self.pix_num_x - 1) / 2.)
        y_coordinate_temp = self.pix_height * (np.array(range(self.pix_num_y)) - (self.pix_num_y - 1) / 2.)
        mesh_temp = np.meshgrid(x_coordinate_temp, y_coordinate_temp)

        self.pixel_position[0, ::, ::, 0] = mesh_temp[0][::, ::]
        self.pixel_position[0, ::, ::, 1] = mesh_temp[1][::, ::]

        # Calculate the index map for the image
        self.pixel_index_map[0, :, :, 0] = mesh_temp[0][::, ::]
        self.pixel_index_map[0, :, :, 1] = mesh_temp[1][::, ::]

        # Initialize the detector effect parameters
        self.pedestal = np.zeros((1, self.pix_num_x, self.pix_num_y))
        self.pixel_rms = np.zeros((1, self.pix_num_x, self.pix_num_y))
        self.pixel_bkgd = np.zeros((1, self.pix_num_x, self.pix_num_y))
        self.pixel_status = np.zeros((1, self.pix_num_x, self.pix_num_y))
        self.pixel_mask = np.zeros((1, self.pix_num_x, self.pix_num_y))
        self.pixel_gain = np.ones((1, self.pix_num_x, self.pix_num_y))

        self._calculate_pixel_range()

    def initialize_as_LCLS_detector(self, path):

        self.mode = "LCLS Detectors"
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
        self.detector_initialized = 1

        # All pre-defined detector are initialized in a similar way.
        # We will parse the path in the self._initialize(path) function again
        self._initialize(path)

        sys.stdout = old_stdout
        f.close()
        os.remove('./Detector_initialization.log')

    def _initialize(self, path):

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
        self.pix_num_x = np.array([self.pixel_index_map.shape[1], ] * self.panel_num)
        self.pix_num_y = np.array([self.pixel_index_map.shape[2], ] * self.panel_num)
        self.pix_num_total = np.sum(np.multiply(self.pix_num_x, self.pix_num_y))

        tmp = float(self.geometry.get_pixel_scale_size())
        self.pix_width = np.ones((self.panel_num, self.pix_num_x[0], self.pix_num_y[0])) * tmp
        self.pix_height = np.ones((self.panel_num, self.pix_num_x[0], self.pix_num_y[0])) * tmp

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

        self._calculate_pixel_range()

    def initialize_pixels_with_beam_information(self, wavevector, polar):

        if self.detector_initialized != 1:
            print("Please initialize the detector with self.initialize_with_predefined_detector_parameters \n" +
                  "or self.initialize_with_detector_geometry_file before you initialize the pixels with beam 'n" +
                  "information.")
            return 1

        (self.pixel_position_reciprocal, self.pixel_distance_reciprocal,
         self.polarization_correction, self.solid_angle_correction) = \
            pysingfel.geometry.reciprocal_space_pixel_position_and_correction(
                pixel_center=self.pixel_position,
                polarization=polar,
                wave_vector=wavevector,
                orientation=self.orientation)

    def print_detector_structure(self):
        if self.mode == "User Defined":
            print("Sorry, currently we can not provide information summary" +
                  " about user defined detectors in standard format.")
        else:
            self.geometry.get_dict_of_comments()
            self.geometry.print_list_of_geos()

    def _calculate_pixel_range(self):
        self.pixel_index_x_max = np.max(self.pixel_index_map[:, :, :, 0]) + 1
        self.pixel_index_y_max = np.max(self.pixel_index_map[:, :, :, 1]) + 1

    def convert_image_stack_to_2d_image(self, image_stack):
        # construct the image holder:
        image = np.zeros((self.pixel_index_x_max, self.pixel_index_y_max))
        for l in range(self.panel_num):
            image[self.pixel_index_map[l, :, :, 0],
                  self.pixel_index_map[l, :, :, 1]] = image_stack[l, :, :]

        return image

    def preferred_reciprocal_mesh_number(self, wave_vector):
        """ Return the prefered the reciprocal voxel grid number along 1 dimension. """
        voxel_length = self.preferred_voxel_length(wave_vector)
        reciprocal_space_range = np.max(self.pixel_distance_reciprocal)
        # The voxel number along 1 dimension is 2*voxel_half_num_1d+1 
        voxel_half_num_1d = int(np.floor_divide(reciprocal_space_range, voxel_length * (1e-10 / 2.)) + 1)

        voxel_num_1d = int(2 * voxel_half_num_1d + 1)
        return voxel_num_1d

    def preferred_voxel_length(self, wave_vector):
        # Notice that the voxel length has something to do with the units which I have never figured out what happened.
        voxel_length = np.sqrt(np.sum(np.square(wave_vector
                                                ))) / self.distance * np.min(
            np.minimum(self.pix_width, self.pix_height))
        return voxel_length

    def get_reciprocal_mesh(self, voxel_number_1d):
        voxel_half_number = int((voxel_number_1d / 2) - 1)
        voxel_length = np.max(self.pixel_distance_reciprocal) / voxel_half_number / (1e-10 / 2)

        voxel_number = int(voxel_number_1d)
        return pysingfel.geometry.get_reciprocal_mesh(voxel_number, voxel_length), voxel_length

    def get_pattern_without_corrections(self, particle, beam, device="cpu"):

        if device == "cpu":
            diffraction_pattern = pd.calculate_molecularFormFactorSq(particle,
                                                                     self.pixel_distance_reciprocal,
                                                                     self.pixel_position_reciprocal)
        elif device == "gpu":
            import pysingfel.gpu.diffraction as pgd
            diffraction_pattern = pgd.calculate_diffraction_pattern_gpu(self.pixel_position_reciprocal,
                                                                        particle,
                                                                        "intensity")
        else:
            print(" The device parameter can only be set as \"gpu\" or \"cpu\" ")
            raise Exception('Wrong parameter value. device can only be set as \"gpu\" or \"cpu\" ')

        return diffraction_pattern * scaling_factor(beam=beam, detector=self)

    #################################################################
    #     From now on, the functions are used to add detector effects
    #################################################################

    def _add_static_noise(self, pattern):
        return pattern + np.random.uniform(0, 2 * np.sqrt(3 * self.pixel_rms))

    def _add_solid_angle_correction(self, pattern):
        return np.multiply(pattern, self.solid_angle_correction)

    def _add_polarization_correction(self, pattern):
        return np.multiply(pattern, self.polarization_correction)

    def add_shot_noise(self, pattern):
        return np.random.poisson(pattern)

    def add_correction_and_quantization(self, pattern):
        return np.random.poisson(np.multiply(np.multiply(pattern,
                                                         self.solid_angle_correction),
                                             self.polarization_correction))

    def add_cross_talk_effect_panel(self, pattern, path):
        return cross_talk_panel(path, pattern)

    def photons_without_static_noise(self, particle, beam, device="cpu"):
        raw_data = self.get_pattern_without_corrections(particle, beam, device)
        return self.add_correction_and_quantization(raw_data)

    def get_ADU(self, particle, beam, path, device="cpu"):
        raw_photon = self.photons_without_static_noise(particle, beam, device)
        return cross_talk_panel(path, raw_photon) + np.random.uniform(0, 2 * np.sqrt(3 * self.pixel_rms))


####################################
#     From now on, the functions are used to add detector effects
####################################

def scaling_factor(beam, detector):
    # Solid angle
    factor = detector.pix_width[0, 0, 0] * detector.pix_height[0, 0, 0] / detector.distance ** 2
    # Thomson factor and intensity
    factor *= 2.81794 * 2.81794 * 1e-30 * beam.get_photonsPerPulse() / (4 * beam.get_focus() ** 2)

    return factor
