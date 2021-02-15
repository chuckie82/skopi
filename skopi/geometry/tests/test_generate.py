import itertools
import numpy as np
import scipy as sp
import pytest

from pysingfel.geometry import generate, convert
import pysingfel.constants as cst


def test_points_on_1sphere_4y():
    """Test points_on_1sphere for 4 points on axis 'y'."""
    points = generate.points_on_1sphere(4, 'y')
    assert np.allclose(points[0], cst.quat1)
    assert np.allclose(points[1], cst.quaty90)
    assert np.allclose(points[2], cst.quaty)


def test_points_on_1sphere_8x():
    """Test points_on_1sphere for 8 points on axis 'x'."""
    points = generate.points_on_1sphere(8, 'x')
    assert np.allclose(points[0], cst.quat1)
    assert np.allclose(points[2], cst.quatx90)
    assert np.allclose(points[4], cst.quatx)


def test_points_on_Nsphere():
    """Test points_on_Nsphere."""
    # Testing a 4-sphere (5D) with 100 points
    points = generate.points_on_Nsphere(100, 4)
    # Norm = 1
    assert np.allclose(sp.linalg.norm(points, axis=1), 1.)


def test_quaternion_product():
    """Test equivalence with generating combined rotations using rotation matrices."""
    qp_x90_y90 = generate.quaternion_product(cst.quatx90, cst.quaty90)
    qp_y90_z90 = generate.quaternion_product(cst.quaty90, cst.quatz90)
    qp_z90_z90 = generate.quaternion_product(cst.quatz90, cst.quatz90)
    
    assert np.allclose(convert.quaternion2rot3d(qp_x90_y90), cst.Rx90.dot(cst.Ry90))
    assert np.allclose(convert.quaternion2rot3d(qp_y90_z90), cst.Ry90.dot(cst.Rz90))
    assert np.allclose(convert.quaternion2rot3d(qp_z90_z90), cst.Rz90.dot(cst.Rz90))
