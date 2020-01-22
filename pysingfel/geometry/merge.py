import numpy as np

from . import convert, mapping


def put_slice(local_index, local_weight, slice_, volume_merge,
              volume_weight):
    """
    Merges one slice to the volume given the index and weight map.

    :param local_index: The index containing values to take.
    :param local_weight: The weight for each index.
    :param slice_: The slice.
    :param volume_merge: The volume to merge to.
    :param volume_weight: The volume to store the weights.
    :return: None.
    """
    pattern_shape = local_index.shape[:3]
    pixel_num = np.prod(pattern_shape)

    volume_shape = volume_merge.shape
    volume_num_1d = volume_shape[0]

    if volume_merge.flags["C_CONTIGUOUS"] and volume_merge.flags["C_CONTIGUOUS"]:
        c_contiguous = True
    elif volume_merge.flags["F_CONTIGUOUS"] and volume_merge.flags["F_CONTIGUOUS"]:
        c_contiguous = False
    else:
        raise AttributeError("Expecting C- or F-contiguous arrays.")

    if c_contiguous:
        convertion_factor = np.array(
            [volume_num_1d**2, volume_num_1d, 1], dtype=np.int64)
        volume_m_1d = volume_merge.ravel(order='C')
        volume_w_1d = volume_weight.ravel(order='C')
    else:
        convertion_factor = np.array(
            [1, volume_num_1d, volume_num_1d**2], dtype=np.int64)
        volume_m_1d = volume_merge.ravel(order='F')
        volume_w_1d = volume_weight.ravel(order='F')

    # Ensure it's a view, not a copy
    assert not volume_m_1d.flags['OWNDATA']
    assert not volume_w_1d.flags['OWNDATA']

    index_2d = np.reshape(local_index, [pixel_num, 8, 3])
    index_2d = np.matmul(index_2d, convertion_factor)

    weight_2d = np.reshape(local_weight, [pixel_num, 8])

    data_1d = np.reshape(slice_, pixel_num)

    # Expand the data to merge
    volume_m_1d[index_2d] += np.multiply(weight_2d, data_1d[:,np.newaxis])
    volume_w_1d[index_2d] += weight_2d


def merge_slice(slice_, pixel_momentum, orientation, volume_merge,
                volume_weight, voxel_length, inverse=False):
    """
    Merge 1 slice.

    :param slice_: The slice.
    :param pixel_momentum: The coordinate of each pixel in the reciprocal space measured in A
    :param orientation: The orientation of the slice.
    :param volume_merge: The volume to merge to.
    :param volume_weight: The volume to store the weights.
    :param voxel_length: The length unit of the voxel
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: None.
    """
    # construct the rotation matrix
    rot_mat = convert.quaternion2rot3d(orientation)
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
        voxel_num_1d=volume_merge.shape[0])
    # get one slice
    return put_slice(local_index=index, local_weight=weight, slice_=slice_,
                     volume_merge=volume_merge, volume_weight=volume_weight)


def merge_slices(slices, pixel_momentum, orientations, volume_num_1d,
                 voxel_length, inverse=False):
    """
    Merge all the slice, given their orientation.

    :param slices: The slice batch.
    :param pixel_momentum: The coordinate of each pixel in the reciprocal space measured in A
    :param orientations: The orientations of the slices.
    :param volume_num_1d: The number of voxel per dimension.
    :param voxel_length: The length unit of the voxel
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: None.
    """
    volume_shape = (volume_num_1d,) *3
    volume_merge = np.zeros(volume_shape)
    volume_weight = np.zeros(volume_shape)

    for l in range(slices.shape[0]):
        merge_slice(slices[l], pixel_momentum, orientations[l],
                    volume_merge, volume_weight, voxel_length, inverse)

    return volume_merge / volume_weight
