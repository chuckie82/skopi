import itertools
import numpy as np
import pytest

from pysingfel import geometry


Rx90 = np.array([
    [1., 0., 0.],
    [0., 0., -1.],
    [0., 1., 0.]])
Ry90 = np.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [-1., 0., 0.]])
Rz90 = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]])


quatx90 = np.array([1., 1., 0., 0.]) / np.sqrt(2)
quaty90 = np.array([1., 0., 1., 0.]) / np.sqrt(2)
quatz90 = np.array([1., 0., 0., 1.]) / np.sqrt(2)


# Replacement test
def test_euler_to_rot3d_equiv_angle_axis_to_rot3d_2():
    """Test equivalence betwen euler_ and angle_axis_ for axis y.

    Show that euler_to_rot3d(0, theta, 0) and
    angle_axis_to_rot3d('y', theta) are equivalent.
    """
    n = 1000
    angles = np.random.rand(n) * 2 * np.pi
    for angle in angles:
        assert np.allclose(
            geometry.euler_to_rot3d(0, angle, 0),
            geometry.angle_axis_to_rot3d('y', angle))


# Replacement test
def test_euler_to_quaternion_equiv_angle_axis_to_quaternion_2():
    """Test equivalence betwen euler_ and angle_axis_ for axis y.

    Show that euler_to_quaternion(0, theta, 0) and
    angle_axis_to_quaternion('y', theta) are equivalent.
    Show that euler_to_quaternion(0, 0, phi) and
    angle_axis_to_quaternion('x', phi) are equivalent.
    """
    n = 1000
    angles = np.random.rand(n) * 2 * np.pi
    for angle in angles:
        assert np.allclose(
            geometry.euler_to_quaternion(0, angle, 0),
            geometry.angle_axis_to_quaternion('y', angle))
        assert np.allclose(
            geometry.euler_to_quaternion(0, 0, angle),
            geometry.angle_axis_to_quaternion('x', angle))
