import numpy as np
from numba import jit
from six import string_types

from skopi.util import deprecated


######################################################################
# Some trivial geometry calculations
######################################################################

# Converters between different descriptions of 3D rotation.
def angle_axis_to_rot3d(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.

    :param axis: A numpy array for the rotation axis.
        Axis names 'x', 'y', and 'z' are also accepted.
    :param theta: Rotation angle.
    :return:
    """
    if isinstance(axis, string_types):
        axis = axis.lower()
        if axis == 'x':
            axis = np.array([1., 0., 0.])
        elif axis == 'y':
            axis = np.array([0., 1., 0.])
        elif axis == 'z':
            axis = np.array([0., 0., 1.])
        else:
            raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
    elif len(axis) != 3:
        raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
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
    rot3d = np.array(
        [[a * a_bracket + cos_theta, a * b_bracket - c_sin_theta, a * c_bracket + b_sin_theta],
         [b * a_bracket + c_sin_theta, b * b_bracket + cos_theta, b * c_bracket - a_sin_theta],
         [c * a_bracket - b_sin_theta, c * b_bracket + a_sin_theta, c * c_bracket + cos_theta]])
    return rot3d


def angle_axis_to_quaternion(axis, theta):
    """
    Convert rotation with angle around an axis to a quaternion.

    :param axis: A numpy array for the rotation axis.
        Axis names 'x', 'y', and 'z' are also accepted.
    :param theta: Rotation angle.
    :return:
    """
    if isinstance(axis, string_types):
        axis = axis.lower()
        if axis == 'x':
            axis = np.array([1., 0., 0.])
        elif axis == 'y':
            axis = np.array([0., 1., 0.])
        elif axis == 'z':
            axis = np.array([0., 0., 1.])
        else:
            raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
    elif len(axis) != 3:
        raise ValueError("Axis should be 'x', 'y', 'z' or a 3D vector.")
    axis /= np.linalg.norm(axis)
    quat = np.zeros(4)
    angle = theta/2
    quat[0] = np.cos(angle)
    quat[1:] = np.sin(angle) * axis

    return quat


@deprecated("Euler angles conventions are used inconsistently "
    "and might be removed in the future. "
    "Please consider another method.")
def euler_to_rot3d(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to a rotation
    matrix in 3D, following a Body 3-2-3 sequence.

    :param psi:
    :param theta:
    :param phi:
    :return:
    """
    rphi = np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    rtheta = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    rpsi = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return np.dot(rpsi, np.dot(rtheta, rphi))


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
def rotmat_to_quaternion(rotmat):
    """
    Convert the rotation matrix to a quaternion.

    This function is adopted form
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    :param rotmat:
    :return:
    """
    r00 = rotmat[0,0]
    r01 = rotmat[0,1]
    r02 = rotmat[0,2]
    r10 = rotmat[1,0]
    r11 = rotmat[1,1]
    r12 = rotmat[1,2]
    r20 = rotmat[2,0]
    r21 = rotmat[2,1]
    r22 = rotmat[2,2]

    tr = r00 + r11 + r22
    quat = np.zeros(4)
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2.   # S=4*qw
        quat[0] = 0.25 * S
        quat[1] = (r21 - r12) / S
        quat[2] = (r02 - r20) / S
        quat[3] = (r10 - r01) / S
    elif (r00 > r11) and (r00 > r22):
        S = np.sqrt(1.0 + r00 - r11 - r22) * 2. # S=4*qx
        quat[0] = (r21 - r12) / S
        quat[1] = 0.25 * S
        quat[2] = (r01 + r10) / S
        quat[3] = (r02 + r20) / S
    elif r11 > r22:
        S = np.sqrt(1.0 + r11 - r00 - r22) * 2. # S=4*qy
        quat[0] = (r02 - r20) / S
        quat[1] = (r01 + r10) / S
        quat[2] = 0.25 * S
        quat[3] = (r12 + r21) / S
    else:
        S = np.sqrt(1.0 + r22 - r00 - r11) * 2. # S=4*qz
        quat[0] = (r10 - r01) / S
        quat[1] = (r02 + r20) / S
        quat[2] = (r12 + r21) / S
        quat[3] = 0.25 * S

    return quat


@jit
def quaternion2rot3d(quat):
    """
    Convert the quaternion to rotation matrix.

    This function was originally adopted from
    https://github.com/duaneloh/Dragonfly/blob/master/src/interp.c
    It has been modified from the original.

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
    rotation[0, 1] = 2. * (q12 - q03)
    rotation[0, 2] = 2. * (q13 + q02)
    rotation[1, 0] = 2. * (q12 + q03)
    rotation[1, 1] = (1. - 2. * (q11 + q33))
    rotation[1, 2] = 2. * (q23 - q01)
    rotation[2, 0] = 2. * (q13 - q02)
    rotation[2, 1] = 2. * (q23 + q01)
    rotation[2, 2] = (1. - 2. * (q11 + q22))

    return rotation
