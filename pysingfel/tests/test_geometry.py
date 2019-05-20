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


def test_angle_axis_to_rot3d_x():
    axis = np.array([1., 0., 0.])
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Rx90)


def test_angle_axis_to_rot3d_y():
    axis = np.array([0., 1., 0.])
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d(axis, theta)
    assert np.allclose(rot90, Ry90)


def test_angle_axis_to_rot3d_invariant():
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
    quat = np.array([1., 1., 0., 0.]) / np.sqrt(2)
    rot90 = geometry.quaternion2rot3d(quat)
    assert np.allclose(rot90, Rx90)


def test_quaternion2rot3d_y():
    quat = np.array([1., 0., 1., 0.]) / np.sqrt(2)
    rot90 = geometry.quaternion2rot3d(quat)
    assert np.allclose(rot90, Ry90)


def test_quaternion2rot3d_invariant():
    n = 1000
    orientations = geometry.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        rot = geometry.quaternion2rot3d(orientation)
        rotated = np.dot(rot, orientation[1:])
        assert np.allclose(rotated, orientation[1:])


def test_rotmat_to_quaternion_x():
    quat = geometry.rotmat_to_quaternion(Rx90)
    assert np.allclose(quat,
        np.array([1., 1., 0., 0.])/np.sqrt(2))


def test_rotmat_to_quaternion_y():
    quat = geometry.rotmat_to_quaternion(Ry90)
    assert np.allclose(quat,
        np.array([1., 0., 1., 0.])/np.sqrt(2))


def test_quat2rot2quat():
    #n = 1000
    #orientations = geometry.get_random_quat(n)
    sqrt22 = np.sqrt(2)/2
    n = 1
    orientations = np.array([
        [sqrt22, sqrt22, 0., 0.]])
    for i in range(n):
        orientation = orientations[i]
        rot = geometry.quaternion2rot3d(orientation)
        quat = geometry.rotmat_to_quaternion(rot)
        assert np.allclose(orientation, quat)
