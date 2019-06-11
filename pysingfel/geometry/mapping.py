import numpy as np
import time
from numba import jit

from pysingfel.util import deprecated


######################################################################
# The following functions are utilized to rotate the pixels in reciprocal space
######################################################################

# @jit(nopython=True, parallel=True)
def rotate_pixels_in_reciprocal_space(rot_mat, pixels_position):
    """
    Rotate the pixel positions according to the rotation matrix

    Note that for np.dot(a,b)
    If a is an N-D array and b is an M-D array (where M>=2),
    it is a sum product over the last axis of a and the second-to-last axis of b.

    :param rot_mat: The rotation matrix for M v
    :param pixels_position: [the other dimensions,  3 for x,y,z]
    :return: np.dot(pixels_position, rot_mat.T)
    """

    return np.dot(pixels_position, rot_mat.T)


######################################################################
# Take slice from the volume
######################################################################

# @jit(nopython=True, parallel=True)
def get_weight_and_index(pixel_position, voxel_length, voxel_num_1d):
    """
    Obtain the weight of the pixel for adjacent voxels.
    In this function, pixel position is first cast to the shape [pixel number,3].

    :param pixel_position: The position of each pixel in the space.
    :param voxel_length:
    :param voxel_num_1d:
    :return:
    """

    # Extract the detector shape
    detector_shape = pixel_position.shape[:-1]
    pixel_num = np.prod(detector_shape)

    # Cast the position infor to the shape [pixel number, 3]
    pixel_position_1d = np.reshape(pixel_position, (pixel_num, 3))

    # convert_to_voxel_unit
    pixel_position_1d_voxel_unit = pixel_position_1d / voxel_length

    # shift the center position
    shift = (voxel_num_1d - 1) / 2
    pixel_position_1d_voxel_unit += shift

    # Get one nearest neighbor
    tmp_index = np.floor(pixel_position_1d_voxel_unit).astype(np.int64)

    # Generate the holders
    indexes = np.zeros((pixel_num, 8, 3), dtype=np.int64)
    weight = np.ones((pixel_num, 8), dtype=np.float64)

    # Calculate the floors and the ceilings
    dfloor = pixel_position_1d_voxel_unit - tmp_index
    dceiling = 1 - dfloor

    # Assign the correct values to the indexes
    indexes[:, 0, :] = tmp_index

    indexes[:, 1, 0] = tmp_index[:, 0]
    indexes[:, 1, 1] = tmp_index[:, 1]
    indexes[:, 1, 2] = tmp_index[:, 2] + 1

    indexes[:, 2, 0] = tmp_index[:, 0]
    indexes[:, 2, 1] = tmp_index[:, 1] + 1
    indexes[:, 2, 2] = tmp_index[:, 2]

    indexes[:, 3, 0] = tmp_index[:, 0]
    indexes[:, 3, 1] = tmp_index[:, 1] + 1
    indexes[:, 3, 2] = tmp_index[:, 2] + 1

    indexes[:, 4, 0] = tmp_index[:, 0] + 1
    indexes[:, 4, 1] = tmp_index[:, 1]
    indexes[:, 4, 2] = tmp_index[:, 2]

    indexes[:, 5, 0] = tmp_index[:, 0] + 1
    indexes[:, 5, 1] = tmp_index[:, 1]
    indexes[:, 5, 2] = tmp_index[:, 2] + 1

    indexes[:, 6, 0] = tmp_index[:, 0] + 1
    indexes[:, 6, 1] = tmp_index[:, 1] + 1
    indexes[:, 6, 2] = tmp_index[:, 2]

    indexes[:, 7, :] = tmp_index + 1

    # Assign the correct values to the weight
    weight[:, 0] = np.prod(dceiling, axis=-1)
    weight[:, 1] = dceiling[:, 0] * dceiling[:, 1] * dfloor[:, 2]
    weight[:, 2] = dceiling[:, 0] * dfloor[:, 1] * dceiling[:, 2]
    weight[:, 3] = dceiling[:, 0] * dfloor[:, 1] * dfloor[:, 2]
    weight[:, 4] = dfloor[:, 0] * dceiling[:, 1] * dceiling[:, 2]
    weight[:, 5] = dfloor[:, 0] * dceiling[:, 1] * dfloor[:, 2]
    weight[:, 6] = dfloor[:, 0] * dfloor[:, 1] * dceiling[:, 2]
    weight[:, 7] = np.prod(dfloor, axis=-1)

    # Change the shape of the index and weight variable
    indexes = np.reshape(indexes, detector_shape + (8, 3))
    weight = np.reshape(weight, detector_shape + (8,))

    return indexes, weight


######################################################################
# Take slice from the volume
######################################################################

# @jit(nopython=True, parallel=True)
def get_weight_in_reciprocal_space(pixel_position, voxel_length, voxel_num_1d):
    """
    Obtain the weight of the pixel for adjacent voxels.
    :param pixel_position: The position of each pixel in the reciprocal space in
    :param voxel_length:
    :param voxel_num_1d:
    :return:
    """
    shift = (voxel_num_1d - 1) / 2
    # convert_to_voxel_unit
    pixel_position_voxel_unit = pixel_position / voxel_length + shift

    # Get the indexes of the eight nearest points.
    num_panel, num_x, num_y, _ = pixel_position.shape

    # Get one nearest neighbor
    tmp_index = np.floor(pixel_position_voxel_unit).astype(np.int64)

    # Generate the holders
    indexes = np.zeros((num_panel, num_x, num_y, 8, 3), dtype=np.int64)
    weight = np.ones((num_panel, num_x, num_y, 8), dtype=np.float64)

    # Calculate the floors and the ceilings
    dfloor = pixel_position_voxel_unit - tmp_index
    dceiling = 1 - dfloor

    # Assign the correct values to the indexes
    indexes[:, :, :, 0, :] = tmp_index

    indexes[:, :, :, 1, 0] = tmp_index[:, :, :, 0]
    indexes[:, :, :, 1, 1] = tmp_index[:, :, :, 1]
    indexes[:, :, :, 1, 2] = tmp_index[:, :, :, 2] + 1

    indexes[:, :, :, 2, 0] = tmp_index[:, :, :, 0]
    indexes[:, :, :, 2, 1] = tmp_index[:, :, :, 1] + 1
    indexes[:, :, :, 2, 2] = tmp_index[:, :, :, 2]

    indexes[:, :, :, 3, 0] = tmp_index[:, :, :, 0]
    indexes[:, :, :, 3, 1] = tmp_index[:, :, :, 1] + 1
    indexes[:, :, :, 3, 2] = tmp_index[:, :, :, 2] + 1

    indexes[:, :, :, 4, 0] = tmp_index[:, :, :, 0] + 1
    indexes[:, :, :, 4, 1] = tmp_index[:, :, :, 1]
    indexes[:, :, :, 4, 2] = tmp_index[:, :, :, 2]

    indexes[:, :, :, 5, 0] = tmp_index[:, :, :, 0] + 1
    indexes[:, :, :, 5, 1] = tmp_index[:, :, :, 1]
    indexes[:, :, :, 5, 2] = tmp_index[:, :, :, 2] + 1

    indexes[:, :, :, 6, 0] = tmp_index[:, :, :, 0] + 1
    indexes[:, :, :, 6, 1] = tmp_index[:, :, :, 1] + 1
    indexes[:, :, :, 6, 2] = tmp_index[:, :, :, 2]

    indexes[:, :, :, 7, :] = tmp_index + 1

    # Assign the correct values to the weight
    weight[:, :, :, 0] = np.prod(dceiling, axis=-1)
    weight[:, :, :, 1] = dceiling[:, :, :, 0] * dceiling[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 2] = dceiling[:, :, :, 0] * dfloor[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 3] = dceiling[:, :, :, 0] * dfloor[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 4] = dfloor[:, :, :, 0] * dceiling[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 5] = dfloor[:, :, :, 0] * dceiling[:, :, :, 1] * dfloor[:, :, :, 2]
    weight[:, :, :, 6] = dfloor[:, :, :, 0] * dfloor[:, :, :, 1] * dceiling[:, :, :, 2]
    weight[:, :, :, 7] = np.prod(dfloor, axis=-1)

    return indexes, weight
