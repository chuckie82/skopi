import numpy as np
import time
from numba import jit

from . import convert, mapping


def take_one_slice(local_index, local_weight, volume, pixel_num, pattern_shape):
    """
    Take one slice from the volume given the index and weight and some
    other information.

    :param local_index: The index containing values to take.
    :param local_weight: The weight for each index
    :param volume: The volume to slice from
    :param pixel_num: pixel number.
    :param pattern_shape: The shape of the pattern
    :return: The slice.
    """
    # Convert the index of the 3D diffraction volume to 1D
    volume_num_1d = volume.shape[0]
    convertion_factor = np.array([volume_num_1d * volume_num_1d, volume_num_1d, 1], dtype=np.int64)

    index_2d = np.reshape(local_index, [pixel_num, 8, 3])
    index_2d = np.matmul(index_2d, convertion_factor)

    volume_1d = np.reshape(volume, volume_num_1d ** 3)
    weight_2d = np.reshape(local_weight, [pixel_num, 8])

    # Expand the data to merge
    data_to_merge = volume_1d[index_2d]

    # Merge the data
    data_merged = np.sum(np.multiply(weight_2d, data_to_merge), axis=-1)

    return np.reshape(data_merged, pattern_shape)


def take_n_slice(pattern_shape, pixel_momentum,
                 volume, voxel_length, orientations, inverse=False):
    """
    Take several different slices.

    :param pattern_shape: The shape of the pattern.
    :param pixel_momentum: The coordinate of each pixel in the reciprocal space measured in A
    :param volume: The volume to slice from.
    :param voxel_length: The length unit of the voxel
    :param orientations: The orientation of the slices.
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: n slices.
    """
    # Preprocess
    slice_num = orientations.shape[0]
    pixel_num = np.prod(pattern_shape)

    # Create variable to hold the slices
    slices_holder = np.zeros((slice_num,) + tuple(pattern_shape))

    tic = time.time()
    for l in range(slice_num):
        # construct the rotation matrix
        rot_mat = convert.quaternion2rot3d(orientations[l, :])
        if inverse:
            rot_mat = np.linalg.inv(rot_mat)

        # rotate the pixels in the reciprocal space.
        # Notice that at this time, the pixel position is in 3D
        rotated_pixel_position = mapping.rotate_pixels_in_reciprocal_space(
            rot_mat, pixel_momentum)
        # calculate the index and weight in 3D
        index, weight = mapping.get_weight_and_index(
            pixel_position=rotated_pixel_position,
            voxel_length=voxel_length,
            voxel_num_1d=volume.shape[0])
        # get one slice
        slices_holder[l] = take_one_slice(local_index=index,
                                          local_weight=weight,
                                          volume=volume,
                                          pixel_num=pixel_num,
                                          pattern_shape=pattern_shape)

    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds" % (slice_num, toc - tic))

    return slices_holder
