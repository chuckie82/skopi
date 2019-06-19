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



def test_get_reciprocal_mesh_shape_even():
    """Test get_reciprocal_mesh shape for even number."""
    voxel_num = 8
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert mesh.shape == (voxel_num,) * 3 + (3,)


def test_get_reciprocal_mesh_shape_odd():
    """Test get_reciprocal_mesh shape for odd number."""
    voxel_num = 9
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert mesh.shape == (voxel_num,) * 3 + (3,)


def test_get_reciprocal_mesh_step_even():
    """Test get_reciprocal_mesh step for even number."""
    voxel_num = 8
    i = 4
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert np.isclose(mesh[i, i, i, 0] - mesh[i-1, i-1, i-1, 0], voxel_length)
    assert np.isclose(mesh[i, i, i, 1] - mesh[i-1, i-1, i-1, 1], voxel_length)
    assert np.isclose(mesh[i, i, i, 2] - mesh[i-1, i-1, i-1, 2], voxel_length)


def test_get_reciprocal_mesh_step_odd():
    """Test get_reciprocal_mesh step for odd number."""
    voxel_num = 9
    i = 4
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert np.isclose(mesh[i, i, i, 0] - mesh[i-1, i-1, i-1, 0], voxel_length)
    assert np.isclose(mesh[i, i, i, 1] - mesh[i-1, i-1, i-1, 1], voxel_length)
    assert np.isclose(mesh[i, i, i, 2] - mesh[i-1, i-1, i-1, 2], voxel_length)


def test_get_reciprocal_mesh_center_even():
    """Test get_reciprocal_mesh centering for even number."""
    voxel_num = 8
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert np.isclose(mesh[0, 0, 0, 0] + mesh[-1, -1, -1, 0], 0.)
    assert np.isclose(mesh[0, 0, 0, 1] + mesh[-1, -1, -1, 1], 0.)
    assert np.isclose(mesh[0, 0, 0, 2] + mesh[-1, -1, -1, 2], 0.)


def test_get_reciprocal_mesh_center_odd():
    """Test get_reciprocal_mesh centering for odd number."""
    voxel_num = 9
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert np.isclose(mesh[0, 0, 0, 0] + mesh[-1, -1, -1, 0], 0.)
    assert np.isclose(mesh[0, 0, 0, 1] + mesh[-1, -1, -1, 1], 0.)
    assert np.isclose(mesh[0, 0, 0, 2] + mesh[-1, -1, -1, 2], 0.)


def test_get_reciprocal_mesh_orientation():
    """Test get_reciprocal_mesh centering for x, y, z orientation."""
    voxel_num = 9
    mesh, voxel_length = geometry.get_reciprocal_mesh(voxel_num, 1.)
    assert np.all(mesh[0, :, :, 0] < mesh[-1, :, :, 0])
    assert np.all(mesh[:, 0, :, 1] < mesh[:, -1, :, 1])
    assert np.all(mesh[:, :, 0, 2] < mesh[:, :, -1, 2])


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
