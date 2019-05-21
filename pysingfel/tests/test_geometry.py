import numpy as np

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


def test_angle_axis_to_rot3d_x():
    """Test angle_axis_to_rot3d for 90deg rotations along x."""
    axis = np.array([1., 0., 0.])
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Rx90)


def test_angle_axis_to_rot3d_y():
    """Test angle_axis_to_rot3d for 90deg rotations along y."""
    axis = np.array([0., 1., 0.])
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Ry90)


def test_angle_axis_to_rot3d_z():
    """Test angle_axis_to_rot3d for 90deg rotations along z."""
    axis = np.array([0., 0., 1.])
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Rz90)


def test_angle_axis_to_rot3d_invariant():
    """Test the invariance of angle_axis_to_rot3d.

    Test the invariance property of the rotation axis on randomly
    selected axes.
    """
    n = 1000
    orientations = geometry.get_random_quat(n)
    thetas = np.random.rand(n) * 2 * np.pi
    for i in range(n):
        orientation = orientations[i, 1:]
        theta = thetas[i]
        rot = geometry.angle_axis_to_rot3d(orientation, theta)
        rotated = np.dot(rot, orientation)
        assert np.allclose(rotated, orientation)


def test_quaternion2rot3d_x():
    """Test quaternion2rot3d for 90deg rotations along x."""
    rot90 = geometry.quaternion2rot3d(quatx90)
    assert np.allclose(rot90, Rx90)


def test_quaternion2rot3d_y():
    """Test quaternion2rot3d for 90deg rotations along y."""
    rot90 = geometry.quaternion2rot3d(quaty90)
    assert np.allclose(rot90, Ry90)


def test_quaternion2rot3d_z():
    """Test quaternion2rot3d for 90deg rotations along z."""
    rot90 = geometry.quaternion2rot3d(quatz90)
    assert np.allclose(rot90, Rz90)


def test_quaternion2rot3d_invariant():
    """Test the invariance of quaternion2rot3d.

    Test the invariance property of the rotation axis on randomly
    selected axes.
    """
    n = 1000
    orientations = geometry.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        rot = geometry.quaternion2rot3d(orientation)
        rotated = np.dot(rot, orientation[1:])
        assert np.allclose(rotated, orientation[1:])


def test_rotmat_to_quaternion_x():
    """Test rotmat_to_quaternion for 90deg rotations along x."""
    quat = geometry.rotmat_to_quaternion(Rx90)
    assert np.allclose(quat, quatx90)


def test_rotmat_to_quaternion_y():
    """Test rotmat_to_quaternion for 90deg rotations along y."""
    quat = geometry.rotmat_to_quaternion(Ry90)
    assert np.allclose(quat, quaty90)


def test_rotmat_to_quaternion_z():
    """Test rotmat_to_quaternion for 90deg rotations along z."""
    quat = geometry.rotmat_to_quaternion(Rz90)
    assert np.allclose(quat, quatz90)


def test_quat2rot2quat():
    """Test quaternion2rot3d and rotmat_to_quaternion consistency."""
    n = 1000
    orientations = geometry.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        rot = geometry.quaternion2rot3d(orientation)
        quat = geometry.rotmat_to_quaternion(rot)
        assert np.allclose(orientation, quat) \
            or np.allclose(orientation, -quat)
        # quaternions doulbe-cover 3D rotations
