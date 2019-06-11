import itertools
import numpy as np
import pytest

from pysingfel.geometry import generate


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


def test_points_on_1sphere_4y():
    """Test points_on_1sphere for 4 points on axis 'y'."""
    points = generate.points_on_1sphere(4, 'y')
    assert np.allclose(points[0], np.array([1., 0., 0., 0.]))
    assert np.allclose(points[1], quaty90)
    assert np.allclose(points[2], np.array([0., 0., 1., 0.]))


def test_points_on_1sphere_8x():
    """Test points_on_1sphere for 8 points on axis 'x'."""
    points = generate.points_on_1sphere(8, 'x')
    assert np.allclose(points[0], np.array([1., 0., 0., 0.]))
    assert np.allclose(points[2], quatx90)
    assert np.allclose(points[4], np.array([0., 1., 0., 0.]))


class TestPointsOn2Sphere(object):
    """Test points_on_2sphere."""
    @classmethod
    def setup_class(cls):
        cls.n_points = 1000
        cls.points = generate.points_on_2sphere(cls.n_points)

    def _test_points_on_2sphere_moment_1(self, points, thresh):
        """Utility to test points_on_2sphere for moments of order 1."""
        # Normalize real part to be positive because of double-cover
        for i in range(self.n_points):
            points[i] *= np.sign(points[i,0])
        W, X, Y, Z = points.mean(axis=0)
        assert abs(X) < thresh
        assert abs(Y) < thresh
        assert abs(Z) < thresh

    def test_points_on_2sphere_moment_1(self):
        """Test points_on_2sphere for moments of order 1."""
        thresh = 2.3e-3  # Lower limit for original implementation (1000)
        points = self.points.copy()
        self._test_points_on_2sphere_moment_1(points, thresh)
        with pytest.raises(AssertionError):
            self._test_points_on_2sphere_moment_1(points, thresh*.95)

    def _test_points_on_2sphere_moment_2(self, points, thresh_xx, thresh_xy):
        """Utilisty to test points_on_2sphere for moments of order 2."""
        # No need to normalize for double-cover because for even order
        moments_xx = []
        moments_xy = []
        for (i, j) in itertools.product(*(range(4),)*2):
            M2 = np.dot(points[:,i], points[:,j])
            if i == j:
                moments_xx.append(M2)
            else:
                moments_xy.append(M2)
        moments_xx = np.array(moments_xx)
        assert np.all(np.abs(moments_xx-moments_xx.mean()) < thresh_xx)
        assert np.all(np.abs(moments_xy) < thresh_xy)

    def test_points_on_2sphere_moment_2(self):
        """Test points_on_2sphere for moments of order 2."""
        thresh_xx = 0.34  # Lower limit for original implementation (1000)
        thresh_xy = 1.16  # Lower limit for original implementation (1000)
        points = self.points.copy()
        self._test_points_on_2sphere_moment_2(points, thresh_xx, thresh_xy)
        with pytest.raises(AssertionError):
            self._test_points_on_2sphere_moment_2(
                points, thresh_xx*.95, thresh_xy*.95)

    def _test_points_on_2sphere_angle(self, points, thresh):
        """Utility to test points_on_2sphere for angle between elements."""
        dotprod = np.dot(points, points.T)
        assert dotprod.shape == (1000, 1000)
        assert np.allclose(dotprod.diagonal(), 1.)
        np.fill_diagonal(dotprod, 0.)
        min_angle = np.arccos(np.abs(dotprod).max())
        assert min_angle < thresh  # Abs is for double-cover

    def test_points_on_2sphere_angle(self):
        """Test points_on_2sphere for angle between elements."""
        thresh = 0.105  # Lower limit for original implementation (1000)
        points = self.points.copy()
        self._test_points_on_2sphere_angle(points, thresh)
        with pytest.raises(AssertionError):
            self._test_points_on_2sphere_angle(points, thresh*.95)
