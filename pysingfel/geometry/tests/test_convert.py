import itertools
import numpy as np
import pytest

from pysingfel.geometry import convert, generate


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


def test_angle_axis_to_rot3d_x():
    """Test angle_axis_to_rot3d for 90deg rotations along x."""
    axis = np.array([1., 0., 0.])
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Rx90)


def test_angle_axis_to_rot3d_y():
    """Test angle_axis_to_rot3d for 90deg rotations along y."""
    axis = np.array([0., 1., 0.])
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Ry90)


def test_angle_axis_to_rot3d_z():
    """Test angle_axis_to_rot3d for 90deg rotations along z."""
    axis = np.array([0., 0., 1.])
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Rz90)


def test_angle_axis_to_rot3d_x_name():
    """Test angle_axis_to_rot3d for 90deg rotations along x, by name."""
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d('x', theta)
    assert np.allclose(rot90, Rx90)


def test_angle_axis_to_rot3d_y_name():
    """Test angle_axis_to_rot3d for 90deg rotations along y, by name."""
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d('y', theta)
    assert np.allclose(rot90, Ry90)


def test_angle_axis_to_rot3d_z_name():
    """Test angle_axis_to_rot3d for 90deg rotations along z, by name."""
    theta = np.pi/2
    rot90 = convert.angle_axis_to_rot3d('Z', theta)  # Caps should work too
    assert np.allclose(rot90, Rz90)


def test_angle_axis_to_rot3d_invariant():
    """Test the invariance of angle_axis_to_rot3d.

    Test the invariance property of the rotation axis on randomly
    selected axes.
    """
    n = 1000
    orientations = generate.get_random_quat(n)
    thetas = np.random.rand(n) * 2 * np.pi
    for i in range(n):
        orientation = orientations[i, 1:]
        theta = thetas[i]
        rot = convert.angle_axis_to_rot3d(orientation, theta)
        rotated = np.dot(rot, orientation)
        assert np.allclose(rotated, orientation)


def test_angle_axis_to_quaternion_x():
    """Test angle_axis_to_quaternion for 90deg rotations along x."""
    axis = np.array([1., 0., 0.])
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quatx90)


def test_angle_axis_to_quaternion_y():
    """Test angle_axis_to_quaternion for 90deg rotations along y."""
    axis = np.array([0., 1., 0.])
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quaty90)


def test_angle_axis_to_quaternion_z():
    """Test angle_axis_to_quaternion for 90deg rotations along z."""
    axis = np.array([0., 0., 1.])
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quatz90)


def test_angle_axis_to_quaternion_x_name():
    """Test angle_axis_to_quaternion for 90deg rotations along x, by name."""
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion('x', theta)
    assert np.allclose(quat, quatx90)


def test_angle_axis_to_quaternion_y_name():
    """Test angle_axis_to_quaternion for 90deg rotations along y, by name."""
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion('y', theta)
    assert np.allclose(quat, quaty90)


def test_angle_axis_to_quaternion_z_name():
    """Test angle_axis_to_quaternion for 90deg rotations along z, by name."""
    theta = np.pi/2
    quat = convert.angle_axis_to_quaternion('Z', theta)  # Caps should work too
    assert np.allclose(quat, quatz90)


def test_euler_to_rot3d_1():
    """Test euler_to_quaternion for 90deg rotations along 1st axis."""
    rot90 = convert.euler_to_rot3d(np.pi/2, 0., 0.)
    assert np.allclose(rot90, Rz90)


def test_euler_to_rot3d_2():
    """Test euler_to_quaternion for 90deg rotations along 2nd axis."""
    rot90 = convert.euler_to_rot3d(0., np.pi/2, 0.)
    assert np.allclose(rot90, Ry90)


def test_euler_to_rot3d_3():
    """Test euler_to_quaternion for 90deg rotations along 3rd axis."""
    rot90 = convert.euler_to_rot3d(0., 0., np.pi/2)
    assert np.allclose(rot90, Rz90)


def test_euler_to_quaternion_yaw():
    """Test euler_to_quaternion for 90deg rotations along yaw axis."""
    quat = convert.euler_to_quaternion(np.pi/2, 0., 0.)
    assert np.allclose(quat, quatz90)


def test_euler_to_quaternion_pitch():
    """Test euler_to_quaternion for 90deg rotations along pitch axis."""
    quat = convert.euler_to_quaternion(0., np.pi/2, 0.)
    assert np.allclose(quat, quaty90)


def test_euler_to_quaternion_roll():
    """Test euler_to_quaternion for 90deg rotations along roll axis."""
    quat = convert.euler_to_quaternion(0., 0., np.pi/2)
    assert np.allclose(quat, quatx90)


def test_quaternion_to_angle_axis_x():
    """Test quaternion_to_angle_axis for 90deg rotations along x."""
    theta, axis = convert.quaternion_to_angle_axis(quatx90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([1., 0., 0.]))


def test_quaternion_to_angle_axis_y():
    """Test quaternion_to_angle_axis for 90deg rotations along y."""
    theta, axis = convert.quaternion_to_angle_axis(quaty90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([0., 1., 0.]))


def test_quaternion_to_angle_axis_z():
    """Test quaternion_to_angle_axis for 90deg rotations along z."""
    theta, axis = convert.quaternion_to_angle_axis(quatz90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([0., 0., 1.]))


def test_quaternion_to_angle_axis_to_quaternion():
    """Test quaternion_to_angle_axis and reverse for consistency."""
    n = 1000
    orientations = generate.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        theta, axis = convert.quaternion_to_angle_axis(orientation)
        quat = convert.angle_axis_to_quaternion(axis, theta)
        assert np.allclose(orientation, quat) \
            or np.allclose(orientation, -quat)
        # quaternions doulbe-cover 3D rotations


def test_quaternion2rot3d_x():
    """Test quaternion2rot3d for 90deg rotations along x."""
    rot90 = convert.quaternion2rot3d(quatx90)
    assert np.allclose(rot90, Rx90)


def test_quaternion2rot3d_y():
    """Test quaternion2rot3d for 90deg rotations along y."""
    rot90 = convert.quaternion2rot3d(quaty90)
    assert np.allclose(rot90, Ry90)


def test_quaternion2rot3d_z():
    """Test quaternion2rot3d for 90deg rotations along z."""
    rot90 = convert.quaternion2rot3d(quatz90)
    assert np.allclose(rot90, Rz90)


def test_quaternion2rot3d_invariant():
    """Test the invariance of quaternion2rot3d.

    Test the invariance property of the rotation axis on randomly
    selected axes.
    """
    n = 1000
    orientations = generate.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        rot = convert.quaternion2rot3d(orientation)
        rotated = np.dot(rot, orientation[1:])
        assert np.allclose(rotated, orientation[1:])


def test_rotmat_to_quaternion_x():
    """Test rotmat_to_quaternion for 90deg rotations along x."""
    quat = convert.rotmat_to_quaternion(Rx90)
    assert np.allclose(quat, quatx90)


def test_rotmat_to_quaternion_y():
    """Test rotmat_to_quaternion for 90deg rotations along y."""
    quat = convert.rotmat_to_quaternion(Ry90)
    assert np.allclose(quat, quaty90)


def test_rotmat_to_quaternion_z():
    """Test rotmat_to_quaternion for 90deg rotations along z."""
    quat = convert.rotmat_to_quaternion(Rz90)
    assert np.allclose(quat, quatz90)


def test_quat2rot2quat():
    """Test quaternion2rot3d and rotmat_to_quaternion consistency."""
    n = 1000
    orientations = generate.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        rot = convert.quaternion2rot3d(orientation)
        quat = convert.rotmat_to_quaternion(rot)
        assert np.allclose(orientation, quat) \
            or np.allclose(orientation, -quat)
        # quaternions doulbe-cover 3D rotations
