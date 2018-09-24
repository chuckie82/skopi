import numpy as np
import time
from numba import jit
from scipy.stats import special_ortho_group


######################################################################
# The following functions are utilized to rotate the pixels in reciprocal space
######################################################################
#@jit(nopython=True, parallel=True)
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
#@jit(nopython=True, parallel=True)
def get_weight_in_reciprocal_space(pixel_position_reciprocal, voxel_length):
    """
    Obtain the weight of the pixel for adjacent voxels.
    :param pixel_position_reciprocal: The position of each pixel in the reciprocal space in
    :param voxel_length:
    :return:
    """
    # convert_to_voxel_unit
    pixel_position_voxel_unit = pixel_position_reciprocal / voxel_length

    # Get the indexes of the eight nearest points.
    num_panel, num_x, num_y, _ = pixel_position_reciprocal.shape

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


def take_one_slice(index_, weight_, volume_, pixel_num_):
    """
    Take one slice from the volume given the index and weight and some
    other information.

    :param index_: The index containing values to take.
    :param weight_: The weight for each index
    :param volume_: The volume to slice from
    :param pixel_num_: pixel number.
    :return: The slice.
    """
    # Convert the index of the 3D diffraction volume to 1D
    volume_num_1d_ = volume_.shape[0]
    convertion_factor = np.array([volume_num_1d_ * volume_num_1d_, volume_num_1d_, 1], dtype=np.int64)

    index_2d_ = np.reshape(index_, [pixel_num_, 8, 3])
    index_2d_ = np.matmul(index_2d_, convertion_factor)

    volume_1d_ = np.reshape(volume_, volume_num_1d_ ** 3)
    weight_2d_ = np.reshape(weight_, [pixel_num_, 8])

    # Expand the data to merge
    data_to_merge_ = volume_1d_[index_2d_]

    # Merge the data
    data_merge_ = np.sum(np.multiply(weight_2d_, data_to_merge_), axis=1)

    return data_merge_.reshape(index_.shape[:3])


def take_n_slice(pattern_shape, pixel_position, volume, voxel_length, orientations, inverse=False):
    """
    Take several different slices.

    :param pattern_shape: The shape of the pattern.
    :param pixel_position: The coordinate of each pixel in the reciprocal space.
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
        rot_mat = quaternion2rot3d(orientations[l, :])
        if inverse:
            rot_mat = np.linalg.inv(rot_mat)

        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        rotated_pixel_position = rotate_pixels_in_reciprocal_space(rot_mat, pixel_position)
        # calculate the index and weight in 3D
        index, weight = get_weight_in_reciprocal_space(rotated_pixel_position, voxel_length)
        # get one slice
        slices_holder[l, :, :, :] = take_one_slice(index_=index, weight_=weight, volume_=volume, pixel_num_=pixel_num)

    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds" % (slice_num, toc - tic))

    return slices_holder


def take_n_random_slices(detector, volume, voxel_length, number):
    """
    Take n slices from n random orientations.
    
    :param detector: The detector object
    :param volume: The volume to slice from
    :param voxel_length: The voxel length of this volume
    :param number: The number of patterns to slice from.
    :return: [number, panel number, panel pixel number x, panel pixel number y]
    """
    # Preprocess
    pattern_shape_ = detector.pixel_rms.shape
    pixel_position_ = detector.pixel_position_reciprocal.copy()

    # Create variable to hold the slices
    slices = np.zeros((number, pattern_shape_[0], pattern_shape_[1], pattern_shape_[2]))

    tic = time.time()
    for l in range(number):
        # construct the rotation matrix
        rotmat = quaternion2rot3d(get_random_rotation(rotation_axis='random'))
        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        pixel_position_new = rotate_pixels_in_reciprocal_space(rotmat, pixel_position_)
        # calculate the index and weight in 3D
        index, weight_ = get_weight_in_reciprocal_space(pixel_position_new, voxel_length)
        # get one slice
        slices[l, :, :, :] = take_one_slice(index, weight_, volume, detector.pix_num_total)

    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds" % (number, toc - tic))

    return slices


######################################################################
# The following functions are utilized to get corrections
######################################################################
def reshape_pixels_position_arrays_to_1d(array):
    """
    Only an abbreviation.

    :param array: The position array.
    :return: Reshaped array.
    """
    array_1d = np.reshape(array, [np.prod(array.shape[:-1]), 3])
    return array_1d


def get_reciprocal_space_pixel_position(pixel_center, wave_vector):
    """
    Obtain the coordinate of each pixel in the reciprocal space.
    :param pixel_center: The coordinate of  the pixel in real space.
    :param wave_vector: The wavevector.
    :return: The array containing the pixel coordinates.
    """
    # reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)

    # Calculate the reciprocal position of each pixel
    wave_vector_norm = np.sqrt(np.sum(np.square(wave_vector)))
    wave_vector_direction = wave_vector / wave_vector_norm

    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d), axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:, np.newaxis]

    pixel_position_reciprocal_1d = wave_vector_norm * (pixel_center_direction - wave_vector_direction)

    # restore the pixels shape
    pixel_position_reciprocal = np.reshape(pixel_position_reciprocal_1d, pixel_center.shape)

    return pixel_position_reciprocal


def get_polarization_correction(pixel_center, polarization):
    """
    Obtain the polarization correction.

    :param pixel_center: The position of each pixel in real space.
    :param polarization: The polarization vector of the incident beam.
    :return: The polarization correction array.
    """
    # reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)

    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d), axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:, np.newaxis]

    # Calculate the polarization correction
    polarization_norm = np.sqrt(np.sum(np.square(polarization)))
    polarization_direction = polarization / polarization_norm

    polarization_correction_1d = np.sum(np.square(np.cross(pixel_center_direction,
                                                           polarization_direction)), axis=1)

    # print polarization_correction_1d.shape

    polarization_correction = np.reshape(polarization_correction_1d, pixel_center.shape[0:-1])

    return polarization_correction


def solid_angle(pixel_center, pixel_area, orientation):
    """
    Calculate the solid angle for each pixel.

    :param pixel_center: The position of each pixel in real space. In pixel stack format.
    :param orientation: The orientation of the detector.
    :param pixel_area: The pixel area for each pixel. In pixel stack format.
    :return: Solid angle of each pixel.
    """

    # Use 1D format
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)
    pixel_center_norm_1d = np.sqrt(np.sum(np.square(pixel_center_1d), axis=-1))
    pixel_area_1d = np.reshape(pixel_area, np.prod(pixel_area.shape))

    # Calculate the direction of each pixel.
    pixel_center_direction_1d = pixel_center_1d / pixel_center_norm_1d[:, np.newaxis]

    # Normalize the orientation vector
    orientation_norm = np.sqrt(np.sum(np.square(orientation)))
    orientation_normalized = orientation / orientation_norm

    # The correction induced by projection which is a factor of cosine.
    cosine_1d = np.abs(np.dot(pixel_center_direction_1d, orientation_normalized))

    # Calculate the solid angle ignoring the projection
    solid_angle_1d = np.divide(pixel_area_1d, np.square(pixel_center_norm_1d))
    solid_angle_1d = np.multiply(cosine_1d, solid_angle_1d)

    # Restore the pixel stack format
    solid_angle_stack = np.reshape(solid_angle_1d, pixel_area.shape)

    return solid_angle_stack


def get_reciprocal_position_and_correction(pixel_position, pixel_area,
                                           wave_vector, polarization, orientation):
    """
    Calculate the pixel positions in reciprocal space and all the related corrections.

    :param pixel_position: The position of the pixel in real space.
    :param wave_vector: The wavevector.
    :param polarization: The polarization vector.
    :param orientation: The normal direction of the detector.
    :param pixel_area: The pixel area for each pixel. In pixel stack format.
    :return: pixel_position_reciprocal, pixel_position_reciprocal_norm, polarization_correction, geometry_correction
    """
    # Calculate the position and distance in reciprocal space
    pixel_position_reciprocal = get_reciprocal_space_pixel_position(pixel_center=pixel_position,
                                                                    wave_vector=wave_vector)
    pixel_position_reciprocal_norm = np.sqrt(np.sum(np.square(pixel_position_reciprocal), axis=-1))

    # Calculate the corrections.
    polarization_correction = get_polarization_correction(pixel_center=pixel_position,
                                                          polarization=polarization)
    solid_angle_array = solid_angle(pixel_center=pixel_position,
                                    pixel_area=pixel_area,
                                    orientation=orientation)

    return pixel_position_reciprocal, pixel_position_reciprocal_norm, polarization_correction, solid_angle_array


######################################################################
# The following functions are utilized to get reciprocal space grid mesh
######################################################################

def get_reciprocal_mesh(voxel_num_1d, voxel_length):
    """
    Get a symmetric reciprocal coordinate mesh.

    :param voxel_num_1d: An positive odd integer.
    :param voxel_length: The length of the voxel.
    :return: The mesh.
    """
    voxel_half_num_1d = (voxel_num_1d - 1) / 2

    x_meshgrid = (np.array(range(voxel_num_1d)) - voxel_half_num_1d) * voxel_length
    reciprocal_mesh_stack = np.meshgrid(x_meshgrid, x_meshgrid, x_meshgrid)

    reciprocal_mesh = np.zeros((voxel_num_1d, voxel_num_1d, voxel_num_1d, 3))
    for l in range(3):
        reciprocal_mesh[:, :, :, l] = reciprocal_mesh_stack[l][:, :, :]

    return reciprocal_mesh


######################################################################
# The following functions are utilized to assemble the images
######################################################################

def assemble_image_stack(image_stack, index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map

    :param image_stack: [panel num, panel pixel num x, panel pixel num y]
    :param index_map: [panel num, panel pixel num x, panel pixel num y]
    :return: 2D pattern
    """
    # get boundary
    index_max_x = np.max(index_map[:, :, :, 0])
    index_max_y = np.max(index_map[:, :, :, 1])
    # set holder
    image = np.zeros((index_max_x, index_max_y))
    # loop through the panels
    for l in range(index_map.shape[0]):
        image[index_map[l, :, :, :]] = image_stack[l, :, :]

    return image


def assemble_image_stack_batch(image_stack, index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map

    :param image_stack: [stack num, panel num, panel pixel num x, panel pixel num y]
    :param index_map: [panel num, panel pixel num x, panel pixel num y]
    :return: [stack num, 2d pattern x, 2d pattern y]
    """
    # get boundary
    index_max_x = np.max(index_map[:, :, :, 0])
    index_max_y = np.max(index_map[:, :, :, 1])
    # get stack number and panel number
    stack_num = image_stack.shape[0]
    panel_num = image_stack.shape[1]

    # set holder
    image = np.zeros((stack_num, index_max_x, index_max_y))

    # loop through the panels
    for l in range(panel_num):
        image[:, index_map[l, :, :, 0], index_map[l, :, :, 1]] = image_stack[:, l, :, :]

    return image


######################################################################
# Some trivial geometry calculations
######################################################################

# Converters between different descriptions of 3D rotation.
def angle_axis_to_rot3d(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.

    :param axis: A numpy array for the rotation axis.
    :param theta: Rotation angle.
    :return:
    """

    if len(axis) is not 3:
        raise ValueError('Number of axis element must be 3!')
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    cos_theta = np.cos(theta)
    bracket = 1 - cos_theta
    a_bracket = a * bracket
    b_bracket = b * bracket
    c_bracket = c * bracket
    sin_theta = np.sin(theta)
    a_sin_theta = a * sin_theta
    b_sin_theta = b * sin_theta
    c_sin_theta = c * sin_theta
    rot3d = np.array([[a * a_bracket + cos_theta, a * b_bracket - c_sin_theta, a * c_bracket + b_sin_theta],
                      [b * a_bracket + c_sin_theta, b * b_bracket + cos_theta, b * c_bracket - a_sin_theta],
                      [c * a_bracket - b_sin_theta, c * b_bracket + a_sin_theta, c * c_bracket + cos_theta]])
    return rot3d


def euler_to_rot3d(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to a rotation matrix in 3D.
    
    :param psi: 
    :param theta: 
    :param phi: 
    :return: 
    """

    rphi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    rtheta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])
    rpsi = np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return np.dot(rpsi, np.dot(rtheta, rphi))


def euler_to_quaternion(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to quaternion description.
    
    :param psi: 
    :param theta: 
    :param phi: 
    :return: 
    """

    if abs(psi) == 0 and abs(theta) == 0 and abs(phi) == 0:
        quaternion = np.array([1., 0., 0., 0.])
    else:
        r = euler_to_rot3d(psi, theta, phi)
        w = np.array([r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0]])
        if w[0] >= 0:
            w /= np.linalg.norm(w)
        else:
            w /= np.linalg.norm(w) * -1
        theta = np.arccos(0.5 * (np.trace(r) - 1))
        cc_is_theta = np.corrcoef(x=r, y=angle_axis_to_rot3d(w, theta))
        cc_is_negtheta = np.corrcoef(x=r, y=angle_axis_to_rot3d(w, -theta))
        if cc_is_negtheta > cc_is_theta:
            theta = -theta
        quaternion = np.array(
            [np.cos(theta / 2.), np.sin(theta / 2.) * w[0], np.sin(theta / 2.) * w[1], np.sin(theta / 2.) * w[2]])
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def quaternion_to_angle_axis(quaternion):
    """
    Convert quaternion to a right hand rotation theta about certain axis.

    :param quaternion:
    :return:  angle, axis
    """

    ha = np.arccos(quaternion[0])
    theta = 2 * ha
    if theta < np.finfo(float).eps:
        theta = 0
        axis = np.array([1, 0, 0])
    else:
        axis = quaternion[[1, 2, 3]] / np.sin(ha)
    return theta, axis


@jit
def quaternion2rot3d(quat):
    """
    Convert the quaternion to rotation matrix.

    This function is adopted from
    https://github.com/duaneloh/Dragonfly/blob/master/src/interp.c

    :param quat: The quaterion.
    :return: The 3D rotation matrix
    """

    q01 = quat[0] * quat[1]
    q02 = quat[0] * quat[2]
    q03 = quat[0] * quat[3]
    q11 = quat[1] * quat[1]
    q12 = quat[1] * quat[2]
    q13 = quat[1] * quat[3]
    q22 = quat[2] * quat[2]
    q23 = quat[2] * quat[3]
    q33 = quat[3] * quat[3]

    # Obtain the rotation matrix
    rotation = np.zeros((3, 3))
    rotation[0, 0] = (1. - 2. * (q22 + q33))
    rotation[0, 1] = 2. * (q12 + q03)
    rotation[0, 2] = 2. * (q13 - q02)
    rotation[1, 0] = 2. * (q12 - q03)
    rotation[1, 1] = (1. - 2. * (q11 + q33))
    rotation[1, 2] = 2. * (q01 + q23)
    rotation[2, 0] = 2. * (q02 + q13)
    rotation[2, 1] = 2. * (q23 - q01)
    rotation[2, 2] = (1. - 2. * (q11 + q22))

    return rotation


# Functions to generate rotations for different cases: uniform(1d), uniform(3d), random.
def points_on_1sphere(num_pts, rotation_axis):
    """
    Given number of points and axis of rotation, distribute evenly on the surface of a 1-sphere (circle).

    :param num_pts: Number of points 
    :param rotation_axis: Rotation axis.
    :return: Quaternion list of shape [number of quaternion, 4]
    """

    points = np.zeros((num_pts, 4))
    inc_ang = 360. / num_pts
    my_ang = 0
    if rotation_axis == 'y':
        for i in range(num_pts):
            points[i, :] = euler_to_quaternion(0, my_ang * np.pi / 180, 0)
            my_ang += inc_ang
    elif rotation_axis == 'z':
        for i in range(num_pts):
            points[i, :] = euler_to_quaternion(0, 0, my_ang * np.pi / 180)
            my_ang += inc_ang
    return points


def points_on_2sphere(num_pts):
    """
    Given number of points, distribute evenly on hyper surface of a 4-sphere.

    
    :param num_pts: Number of points 
    :return: Quaternion list of shape [number of quaternion, 4]
    """

    points = np.zeros((2 * num_pts, 4))
    dim_num = 4
    # Surface area for unit sphere when dim_num is even
    surface_area = dim_num * np.pi ** (dim_num / 2) / (dim_num / 2)
    delta = np.exp(np.log(surface_area / num_pts) / 3)
    iteration = 0
    ind = 0
    max_iter = 1000
    while ind != num_pts and iteration < max_iter:
        ind = 0
        delta_w1 = delta
        w1 = 0.5 * delta_w1
        while w1 < np.pi:
            q0 = np.cos(w1)
            delta_w2 = delta_w1 / np.sin(w1)
            w2 = 0.5 * delta_w2
            while w2 < np.pi:
                q1 = np.sin(w1) * np.cos(w2)
                delta_w3 = delta_w2 / np.sin(w2)
                w3 = 0.5 * delta_w3
                while w3 < 2 * np.pi:
                    q2 = np.sin(w1) * np.sin(w2) * np.cos(w3)
                    q3 = np.sin(w1) * np.sin(w2) * np.sin(w3)
                    points[ind, :] = np.array([q0, q1, q2, q3])
                    ind += 1
                    w3 += delta_w3
                w2 += delta_w2
            w1 += delta_w1
        delta *= np.exp(np.log(float(ind) / num_pts) / 3)
        iteration += 1
    return points[0:num_pts, :]


def get_random_rotation(rotation_axis):
    """
    Generate a random rotation matrix.

    :param rotation_axis: The rotation axis. If it's 'y', then the rotation is around y axis.
                          Otherwise the rotation is totally random.
    :return: A rotation matrix
    """

    if rotation_axis == 'y':
        u = np.random.random() * 2 * np.pi  # random angle between [0, 2pi]
        return euler_to_quaternion(0, u, 0)
    else:
        return special_ortho_group.rvs(3)
