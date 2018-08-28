import numpy as np
import time
from numba import jit


######################################################################
# Take slice from the volume
######################################################################
@jit
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
    _indexes = np.floor(pixel_position_voxel_unit).astype(np.int64)

    # Generate the holders
    indexes = np.zeros((num_panel, num_x, num_y, 8, 3), dtype=np.int64)
    weight = np.ones((num_panel, num_x, num_y, 8), dtype=np.float64)

    # Calculate the floors and the ceilings
    dfloor = pixel_position_voxel_unit - indexes
    dceiling = 1 - dfloor

    # Assign the correct values to the indexes
    indexes[:, :, :, 0, :] = _indexes

    indexes[:, :, :, 1, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 1, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 1, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 2, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 2, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 2, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 3, 0] = _indexes[:, :, :, 0]
    indexes[:, :, :, 3, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 3, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 4, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 4, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 4, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 5, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 5, 1] = _indexes[:, :, :, 1]
    indexes[:, :, :, 5, 2] = _indexes[:, :, :, 2] + 1

    indexes[:, :, :, 6, 0] = _indexes[:, :, :, 0] + 1
    indexes[:, :, :, 6, 1] = _indexes[:, :, :, 1] + 1
    indexes[:, :, :, 6, 2] = _indexes[:, :, :, 2]

    indexes[:, :, :, 7, :] = _indexes + 1

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


def take_n_slice(pattern_shape, pixel_position, volume, voxel_length, orientations):
    """
    Take several different slices.

    :param pattern_shape: The shape of the pattern.
    :param pixel_position: The coordinate of each pixel in the reciprocal space.
    :param volume: The volume to slice from.
    :param voxel_length: The length unit of the voxel
    :param orientations: The orientation of the slices.
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
        rot_mat = quaternion2rot3D(orientations[l, :])
        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        rotated_pixel_position = rotate_pixels_in_reciprocal_space(rot_mat, pixel_position)
        # calculate the index and weight in 3D
        index, weight = get_weight_in_reciprocal_space(rotated_pixel_position, voxel_length)
        # get one slice
        slices_holder[l, :, :, :] = take_one_slice(index_=index, weight_=weight, volume_=volume, pixel_num_=pixel_num)

    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds" % (slice_num, toc - tic))

    return slices_holder


def take_n_random_slices(detector_, volume_, voxel_length_, number_):
    # Preprocess
    pattern_shape_ = detector_.pixel_rms.shape
    pixel_position_ = detector_.pixel_position_reciprocal.copy()

    # Create variable to hold the slices
    slices_ = np.zeros((number_, pattern_shape_[0], pattern_shape_[1], pattern_shape_[2]))

    tic = time.time()
    for l in range(number_):
        # construct the rotation matrix
        rotmat_ = quaternion2rot3D(getRandomRotation('x'))
        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        pixel_position_new = rotate_pixels_in_reciprocal_space(rotmat_, pixel_position_)
        # calculate the index and weight in 3D
        index_, weight_ = get_weight_in_reciprocal_space(pixel_position_new, voxel_length_)
        # get one slice
        slices_[l, :, :, :] = take_one_slice(index_, weight_, volume_, detector_.pix_num_total)

    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds" % (number_, toc - tic))

    return slices_


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


def _reciprocal_space_pixel_position(pixel_center, wave_vector):
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


def _polarization_correction(pixel_center, polarization):
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

    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center), axis=-1))

    # Calculate the direction of each pixel.
    pixel_center_direction = pixel_center / pixel_center_norm[:, np.newaxis]

    # Normalize the orientation vector
    orientation_norm = np.sqrt(np.sum(np.square(orientation)))
    orientation_normalized = orientation / orientation_norm

    # The correction induced by projection which is a factor of cosine.
    cosine = np.abs(np.dot(pixel_center_direction, orientation_normalized))

    # Calculate the solid angle ignoring the projection
    _solid_angle = np.divide(pixel_area, np.square(pixel_center_norm))
    solid_angle_array = np.multiply(cosine, _solid_angle)

    return solid_angle_array


def reciprocal_position_and_correction(pixel_center, pixel_area,
                                       wave_vector, polarization, orientation):
    """
    Calculate the pixel positions in reciprocal space and all the related corrections.

    :param pixel_center: The position of the pixel in real space.
    :param wave_vector: The wavevector.
    :param polarization: The polarization vector.
    :param orientation: The normal direction of the detector.
    :param pixel_area: The pixel area for each pixel. In pixel stack format.
    :return: pixel_position_reciprocal, pixel_position_reciprocal_norm, polarization_correction, geometry_correction
    """
    # Calculate the position and distance in reciprocal space
    pixel_position_reciprocal = _reciprocal_space_pixel_position(pixel_center=pixel_center,
                                                                 wave_vector=wave_vector)
    pixel_position_reciprocal_norm = np.sqrt(np.sum(np.square(pixel_position_reciprocal), axis=-1))

    # Calculate the corrections.
    polarization_correction = _polarization_correction(pixel_center=pixel_center,
                                                       polarization=polarization)
    solid_angle_array = solid_angle(pixel_center=pixel_center,
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

def assemble_image_from_index_and_panel(image_stack, index):
    # get boundary
    index_max_x = np.max(index[:, :, :, 0])
    index_max_y = np.max(index[:, :, :, 1])
    # set holder
    image = np.zeros((index_max_x, index_max_y))
    # loop through the panels
    for l in range(index.shape[0]):
        image[index[l, :, :, :]] = image_stack[l, :, :]

    return image


def batch_assemble_image_from_index_and_panel(image_stack, index):
    pass


######################################################################
# The following functions are utilized to rotate the pixels in reciprocal space
######################################################################

def rotate_pixels_in_reciprocal_space(rot_mat, pixels_position):
    pixels_position_1d = reshape_pixels_position_arrays_to_1d(pixels_position)
    pixels_position_1d = pixels_position_1d.dot(rot_mat)
    return np.reshape(pixels_position_1d, pixels_position.shape)


######################################################################
# Some trivial geometry calculations
######################################################################

def corrCoeff(X, Y):
    x = X.reshape(-1)
    y = Y.reshape(-1)
    x -= np.mean(x)
    y -= np.mean(y)
    return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))


# Converters between different descriptions of 3D rotation.
def angleAxis2rot3D(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.
    """
    if len(axis) is not 3:
        raise ValueError('Number of axis element must be 3!')
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    cosTheta = np.cos(theta)
    bracket = 1 - cosTheta
    aBracket = a * bracket
    bBracket = b * bracket
    cBracket = c * bracket
    sinTheta = np.sin(theta)
    aSinTheta = a * sinTheta
    bSinTheta = b * sinTheta
    cSinTheta = c * sinTheta
    rot3D = np.array([[a * aBracket + cosTheta, a * bBracket - cSinTheta, a * cBracket + bSinTheta],
                      [b * aBracket + cSinTheta, b * bBracket + cosTheta, b * cBracket - aSinTheta],
                      [c * aBracket - bSinTheta, c * bBracket + aSinTheta, c * cBracket + cosTheta]])
    return rot3D


def euler2rot3D(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to a rotation matrix in 3D.
    """
    Rphi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    Rtheta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])
    Rpsi = np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return np.dot(Rpsi, np.dot(Rtheta, Rphi))


def euler2quaternion(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to quaternion description.
    """
    if abs(psi) == 0 and abs(theta) == 0 and abs(phi) == 0:
        quaternion = np.array([1., 0., 0., 0.])
    else:
        R = euler2rot3D(psi, theta, phi)
        W = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
        if W[0] >= 0:
            W /= np.linalg.norm(W)
        else:
            W /= np.linalg.norm(W) * -1
        theta = np.arccos(0.5 * (np.trace(R) - 1))
        CCisTheta = corrCoeff(R, angleAxis2rot3D(W, theta))
        CCisNegTheta = corrCoeff(R, angleAxis2rot3D(W, -theta))
        if CCisNegTheta > CCisTheta:
            theta = -theta
        quaternion = np.array(
            [np.cos(theta / 2.), np.sin(theta / 2.) * W[0], np.sin(theta / 2.) * W[1], np.sin(theta / 2.) * W[2]])
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def quaternion2AngleAxis(quaternion):
    """
    Convert quaternion to a right hand rotation theta about certain axis.
    """
    HA = np.arccos(quaternion[0])
    theta = 2 * HA
    if theta < np.finfo(float).eps:
        theta = 0
        axis = np.array([1, 0, 0])
    else:
        axis = quaternion[[1, 2, 3]] / np.sin(HA)
    return theta, axis


@jit
def quaternion2rot3D(quat):
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
def pointsOn1Sphere(numPts, rotationAxis):
    """
    Given number of points and axis of rotation, distribute evenly on the surface of a 1-sphere (circle).
    """
    points = np.zeros((numPts, 4))
    incAng = 360. / numPts
    myAng = 0
    if rotationAxis == 'y':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, myAng * np.pi / 180, 0)
            myAng += incAng
    elif rotationAxis == 'z':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, 0, myAng * np.pi / 180)
            myAng += incAng
    return points


def pointsOn4Sphere(numPts):
    """
    Given number of points, distribute evenly on hyper surface of a 4-sphere.
    """
    points = np.zeros((2 * numPts, 4))
    N = 4
    surfaceArea = N * np.pi ** (N / 2) / (N / 2)  # for even N
    delta = np.exp(np.log(surfaceArea / numPts) / 3)
    Iter = 0
    ind = 0
    maxIter = 1000
    while ind != numPts and Iter < maxIter:
        ind = 0
        deltaW1 = delta
        w1 = 0.5 * deltaW1
        while w1 < np.pi:
            q0 = np.cos(w1)
            deltaW2 = deltaW1 / np.sin(w1)
            w2 = 0.5 * deltaW2
            while w2 < np.pi:
                q1 = np.sin(w1) * np.cos(w2)
                deltaW3 = deltaW2 / np.sin(w2)
                w3 = 0.5 * deltaW3
                while w3 < 2 * np.pi:
                    q2 = np.sin(w1) * np.sin(w2) * np.cos(w3)
                    q3 = np.sin(w1) * np.sin(w2) * np.sin(w3)
                    points[ind, :] = np.array([q0, q1, q2, q3])
                    ind += 1
                    w3 += deltaW3
                w2 += deltaW2
            w1 += deltaW1
        delta *= np.exp(np.log(float(ind) / numPts) / 3)
        Iter += 1
    return points[0:numPts, :]


def getRandomRotation(rotationAxis):
    """
    Generate random rotation.
    """
    if rotationAxis == 'y':
        u = np.random.random() * 2 * np.pi  # random angle between [0, 2pi]
        return euler2quaternion(0, u, 0)
    else:
        u = np.random.rand(3)  # uniform random distribution in the [0,1] interval
        # generate uniform random quaternion on SO(3)
        return np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]), np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                         np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]), np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
