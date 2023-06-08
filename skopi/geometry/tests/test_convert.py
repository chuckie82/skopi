import itertools
import numpy as np
import pytest

from skopi.geometry import convert, generate
import skopi.constants as cst


def test_angle_axis_to_rot3d_x():
    """Test angle_axis_to_rot3d for 90deg rotations along x."""
    rot90 = convert.angle_axis_to_rot3d(cst.vecx, np.pi/2)
    assert np.allclose(rot90, cst.Rx90)


def test_angle_axis_to_rot3d_y():
    """Test angle_axis_to_rot3d for 90deg rotations along y."""
    rot90 = convert.angle_axis_to_rot3d(cst.vecy, np.pi/2)
    assert np.allclose(rot90, cst.Ry90)


def test_angle_axis_to_rot3d_z():
    """Test angle_axis_to_rot3d for 90deg rotations along z."""
    rot90 = convert.angle_axis_to_rot3d(cst.vecz, np.pi/2)
    assert np.allclose(rot90, cst.Rz90)


def test_angle_axis_to_rot3d_x_name():
    """Test angle_axis_to_rot3d for 90deg rotations along x, by name."""
    rot90 = convert.angle_axis_to_rot3d('x', np.pi/2)
    assert np.allclose(rot90, cst.Rx90)


def test_angle_axis_to_rot3d_y_name():
    """Test angle_axis_to_rot3d for 90deg rotations along y, by name."""
    rot90 = convert.angle_axis_to_rot3d('y', np.pi/2)
    assert np.allclose(rot90, cst.Ry90)


def test_angle_axis_to_rot3d_z_name():
    """Test angle_axis_to_rot3d for 90deg rotations along z, by name."""
    # Caps should work too
    rot90 = convert.angle_axis_to_rot3d('Z', np.pi/2)
    assert np.allclose(rot90, cst.Rz90)


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
    quat = convert.angle_axis_to_quaternion(cst.vecx, np.pi/2)
    assert np.allclose(quat, cst.quatx90)


def test_angle_axis_to_quaternion_y():
    """Test angle_axis_to_quaternion for 90deg rotations along y."""
    quat = convert.angle_axis_to_quaternion(cst.vecy, np.pi/2)
    assert np.allclose(quat, cst.quaty90)


def test_angle_axis_to_quaternion_z():
    """Test angle_axis_to_quaternion for 90deg rotations along z."""
    quat = convert.angle_axis_to_quaternion(cst.vecz, np.pi/2)
    assert np.allclose(quat, cst.quatz90)


def test_angle_axis_to_quaternion_x_name():
    """Test angle_axis_to_quaternion for 90deg rotations along x, by name."""
    quat = convert.angle_axis_to_quaternion('x', np.pi/2)
    assert np.allclose(quat, cst.quatx90)


def test_angle_axis_to_quaternion_y_name():
    """Test angle_axis_to_quaternion for 90deg rotations along y, by name."""
    quat = convert.angle_axis_to_quaternion('y', np.pi/2)
    assert np.allclose(quat, cst.quaty90)


def test_angle_axis_to_quaternion_z_name():
    """Test angle_axis_to_quaternion for 90deg rotations along z, by name."""
    # Caps should work too
    quat = convert.angle_axis_to_quaternion('Z', np.pi/2)
    assert np.allclose(quat, cst.quatz90)


def test_quaternion_to_angle_axis_x():
    """Test quaternion_to_angle_axis for 90deg rotations along x."""
    theta, axis = convert.quaternion_to_angle_axis(cst.quatx90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, cst.vecx)


def test_quaternion_to_angle_axis_y():
    """Test quaternion_to_angle_axis for 90deg rotations along y."""
    theta, axis = convert.quaternion_to_angle_axis(cst.quaty90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, cst.vecy)


def test_quaternion_to_angle_axis_z():
    """Test quaternion_to_angle_axis for 90deg rotations along z."""
    theta, axis = convert.quaternion_to_angle_axis(cst.quatz90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, cst.vecz)


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
    rot90 = convert.quaternion2rot3d(cst.quatx90)
    assert np.allclose(rot90, cst.Rx90)


def test_quaternion2rot3d_y():
    """Test quaternion2rot3d for 90deg rotations along y."""
    rot90 = convert.quaternion2rot3d(cst.quaty90)
    assert np.allclose(rot90, cst.Ry90)


def test_quaternion2rot3d_z():
    """Test quaternion2rot3d for 90deg rotations along z."""
    rot90 = convert.quaternion2rot3d(cst.quatz90)
    assert np.allclose(rot90, cst.Rz90)


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
    quat = convert.rotmat_to_quaternion(cst.Rx90)
    assert np.allclose(quat, cst.quatx90)


def test_rotmat_to_quaternion_y():
    """Test rotmat_to_quaternion for 90deg rotations along y."""
    quat = convert.rotmat_to_quaternion(cst.Ry90)
    assert np.allclose(quat, cst.quaty90)


def test_rotmat_to_quaternion_z():
    """Test rotmat_to_quaternion for 90deg rotations along z."""
    quat = convert.rotmat_to_quaternion(cst.Rz90)
    assert np.allclose(quat, cst.quatz90)


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
