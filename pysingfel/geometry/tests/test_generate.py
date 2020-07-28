import itertools
import numpy as np
import scipy as sp
import pytest

from pysingfel.geometry import generate
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
