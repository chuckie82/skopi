import itertools
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


def test_angle_axis_to_rot3d_x_name():
    """Test angle_axis_to_rot3d for 90deg rotations along x, by name."""
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d('x', theta)
    assert np.allclose(rot90, Rx90)


def test_angle_axis_to_rot3d_y_name():
    """Test angle_axis_to_rot3d for 90deg rotations along y, by name."""
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d('y', theta)
    assert np.allclose(rot90, Ry90)


def test_angle_axis_to_rot3d_z_name():
    """Test angle_axis_to_rot3d for 90deg rotations along z, by name."""
    theta = np.pi/2
    rot90 = geometry.angle_axis_to_rot3d('Z', theta)  # Caps should work too
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


def test_angle_axis_to_quaternion_x():
    """Test angle_axis_to_quaternion for 90deg rotations along x."""
    axis = np.array([1., 0., 0.])
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quatx90)


def test_angle_axis_to_quaternion_y():
    """Test angle_axis_to_quaternion for 90deg rotations along y."""
    axis = np.array([0., 1., 0.])
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quaty90)


def test_angle_axis_to_quaternion_z():
    """Test angle_axis_to_quaternion for 90deg rotations along z."""
    axis = np.array([0., 0., 1.])
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion(axis, theta)
    assert np.allclose(quat, quatz90)


def test_angle_axis_to_quaternion_x_name():
    """Test angle_axis_to_quaternion for 90deg rotations along x, by name."""
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion('x', theta)
    assert np.allclose(quat, quatx90)


def test_angle_axis_to_quaternion_y_name():
    """Test angle_axis_to_quaternion for 90deg rotations along y, by name."""
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion('y', theta)
    assert np.allclose(quat, quaty90)


def test_angle_axis_to_quaternion_z_name():
    """Test angle_axis_to_quaternion for 90deg rotations along z, by name."""
    theta = np.pi/2
    quat = geometry.angle_axis_to_quaternion('Z', theta)  # Caps should work too
    assert np.allclose(quat, quatz90)


def test_euler_to_rot3d_1():
    """Test euler_to_quaternion for 90deg rotations along 1st axis."""
    rot90 = geometry.euler_to_rot3d(np.pi/2, 0., 0.)
    assert np.allclose(rot90, Rz90)


def test_euler_to_rot3d_2():
    """Test euler_to_quaternion for 90deg rotations along 2nd axis."""
    rot90 = geometry.euler_to_rot3d(0., np.pi/2, 0.)
    assert np.allclose(rot90, Ry90)


def test_euler_to_rot3d_3():
    """Test euler_to_quaternion for 90deg rotations along 3rd axis."""
    rot90 = geometry.euler_to_rot3d(0., 0., np.pi/2)
    assert np.allclose(rot90, Rz90)


def test_euler_to_quaternion_yaw():
    """Test euler_to_quaternion for 90deg rotations along yaw axis."""
    quat = geometry.euler_to_quaternion(np.pi/2, 0., 0.)
    assert np.allclose(quat, quatz90)


def test_euler_to_quaternion_pitch():
    """Test euler_to_quaternion for 90deg rotations along pitch axis."""
    quat = geometry.euler_to_quaternion(0., np.pi/2, 0.)
    assert np.allclose(quat, quaty90)


def test_euler_to_quaternion_roll():
    """Test euler_to_quaternion for 90deg rotations along roll axis."""
    quat = geometry.euler_to_quaternion(0., 0., np.pi/2)
    assert np.allclose(quat, quatx90)


def test_quaternion_to_angle_axis_x():
    """Test quaternion_to_angle_axis for 90deg rotations along x."""
    theta, axis = geometry.quaternion_to_angle_axis(quatx90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([1., 0., 0.]))


def test_quaternion_to_angle_axis_y():
    """Test quaternion_to_angle_axis for 90deg rotations along y."""
    theta, axis = geometry.quaternion_to_angle_axis(quaty90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([0., 1., 0.]))


def test_quaternion_to_angle_axis_z():
    """Test quaternion_to_angle_axis for 90deg rotations along z."""
    theta, axis = geometry.quaternion_to_angle_axis(quatz90)
    assert np.isclose(theta, np.pi/2)
    assert np.allclose(axis, np.array([0., 0., 1.]))


def test_quaternion_to_angle_axis_to_quaternion():
    """Test quaternion_to_angle_axis and reverse for consistency."""
    n = 1000
    orientations = geometry.get_random_quat(n)
    for i in range(n):
        orientation = orientations[i]
        theta, axis = geometry.quaternion_to_angle_axis(orientation)
        quat = geometry.angle_axis_to_quaternion(axis, theta)
        assert np.allclose(orientation, quat) \
            or np.allclose(orientation, -quat)
        # quaternions doulbe-cover 3D rotations


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


def test_points_on_1sphere_4y():
    """Test points_on_1sphere for 4 points on axis 'y'."""
    points = geometry.points_on_1sphere(4, 'y')
    assert np.allclose(points[0], np.array([1., 0., 0., 0.]))
    assert np.allclose(points[1], quaty90)
    assert np.allclose(points[2], np.array([0., 0., 1., 0.]))


def test_points_on_1sphere_8x():
    """Test points_on_1sphere for 8 points on axis 'x'."""
    points = geometry.points_on_1sphere(8, 'x')
    assert np.allclose(points[0], np.array([1., 0., 0., 0.]))
    assert np.allclose(points[2], quatx90)
    assert np.allclose(points[4], np.array([0., 1., 0., 0.]))


class TestPointsOn2Sphere(object):
    """Test points_on_2sphere."""
    @classmethod
    def setup_class(cls):
        cls.n_points = 1000
        cls.points = geometry.points_on_2sphere(cls.n_points)

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
        thresh = 0.11  # Lower limit for original implementation (1000)
        points = self.points.copy()
        self._test_points_on_2sphere_angle(points, thresh)


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
