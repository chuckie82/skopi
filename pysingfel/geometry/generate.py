import numpy as np
from numba import jit
from scipy.stats import special_ortho_group

from pysingfel.util import deprecated

from . import convert


# Functions to generate rotations for different cases: uniform(1d), uniform(3d), random.
def points_on_1sphere(num_pts, rotation_axis):
    """
    Distribute points evenly on a 1-sphere (circle) in 4D.

    :param num_pts: Number of points
    :param rotation_axis: Rotation axis.
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    points = np.zeros((num_pts, 4))
    inc_ang = 2 * np.pi / num_pts
    my_ang = 0
    for i in range(num_pts):
        points[i, :] = convert.angle_axis_to_quaternion(rotation_axis, my_ang)
        my_ang += inc_ang
    return points


@deprecated("The function points_on_2sphere actually generates "
    "points on a 3-sphere, in 4D. "
    "Please call points_on_3sphere instead.")
def points_on_2sphere(num_pts):
    return points_on_3sphere(num_pts)


def points_on_3sphere(num_pts):
    """
    Distribute points evenly on a 3-sphere in 4D.

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


def get_random_rotation(rotation_axis=None):
    """
    Generate a random rotation matrix.

    :param rotation_axis: The rotation axis.
        If it's 'x', 'y', or 'z', then the rotation is around that axis.
        Otherwise the rotation is totally random.
    :return: A rotation matrix
    """
    rotation_axis = rotation_axis.lower()
    if rotation_axis in ('x', 'y', 'z'):
        u = np.random.rand() * 2 * np.pi  # random angle between [0, 2pi]
        return convert.angle_axis_to_rot3d(rotation_axis, u)
    else:
        return special_ortho_group.rvs(3)


def get_random_quat(num_pts):
    """
    Get num_pts of unit quaternions on the 4 sphere with a uniform random distribution.

    :param num_pts: The number of quaternions to return
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    u = np.random.rand(3, num_pts)
    u1, u2, u3 = [u[x] for x in range(3)]

    quat = np.zeros((4, num_pts))
    quat[0] = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    quat[1] = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    quat[2] = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    quat[3] = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.transpose(quat)


def get_uniform_quat(num_pts):
    """
    Get num_pts of unit quaternions evenly distributed on the 3-sphere.

    :param num_pts: The number of quaternions to return
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    return points_on_3sphere(num_pts)
