import numpy as np
from scipy.stats import special_ortho_group

from skopi.util import deprecated

from . import convert


# Functions to generate rotations for different cases: uniform(1d), uniform(3d), random.
def points_on_1sphere(num_pts, rotation_axis):
    """
    Distribute points evenly on a unit 1-sphere (circle) in 4D.

    :param num_pts: Number of points
    :param rotation_axis: Rotation axis.
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    points = np.zeros((num_pts, 4))
    inc_ang = 2 * np.pi / num_pts
    my_ang = 0
    for i in range(num_pts):
        points[i, :] = convert.angle_axis_to_quaternion(rotation_axis, my_ang)
        my_ang += inc_ang
    return points


@deprecated("The function points_on_2sphere actually generates "
    "points on a 3-sphere, in 4D. "
    "Please call points_on_3sphere instead.")
def points_on_2sphere(num_pts):
    return points_on_3sphere(num_pts)


def points_on_3sphere(num_pts):
    """
    Attempt to distribute points evenly on a unit 3-sphere in 4D.

    The 3-sphere is the clasical hypersphere embedded in the 4D
    Euclidian space.

    :param num_pts: Number of points
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    return points_on_Nsphere(num_pts, 3)


def points_on_3hemisphere(num_pts):
    """
    Attempt to distribute points evenly on half of a unit 3-sphere in 4D.

    The 3-sphere is the clasical hypersphere embedded in the 4D
    Euclidian space.

    :param num_pts: Number of points
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    return points_on_Nsphere(num_pts, 3, half=True)


def points_on_Nsphere(num_pts, N, half=False):
    """
    Attempt to distribute points evenly on a unit N-sphere.

    This N-sphere corresponds to the set of points in an N+1 dimensional
    Euclidean space that are at a unit distance of the origin.

    :param num_pts: Number of points
    :param half: Bool. If True, distribute on half the N-sphere.
    :return: List of (N+1)-D points [num_pts, N+1]
    """
    dim_num = N+1

    surface_area = _surface_Nsphere(N)
    if half:
        surface_area /= 2
    delta = np.exp(np.log(surface_area / num_pts) / N)
    iteration = 0
    ind = 0
    max_iter = 1000

    points = np.zeros((2 * num_pts, dim_num))
    best_pts = None
    best_num_points = 2 * num_pts

    while ind != num_pts and iteration < max_iter:
        ind = _point_on_Nsphere_loop(
            points, delta, 0, dim_num, 0, half=half)
        delta *= np.exp(np.log(float(ind) / num_pts) / N)
        iteration += 1

        if num_pts <= ind < best_num_points:
            best_pts = points[:ind].copy()
            best_num_points = ind

    return best_pts[:num_pts]


def _point_on_Nsphere_loop(points, delta, currDim, NDim, ind,
                           base=1, last=1, half=False):
    """Internal and recursive logic of points_on_Nsphere."""
    delta_w = delta / last
    w = 0.5 * delta_w

    if currDim+2 == NDim:
        w_limit = np.pi if half else 2*np.pi
        while w < w_limit - delta_w/2:
            points[ind, currDim] = base * np.cos(w)
            points[ind, currDim+1] = base * np.sin(w)
            ind += 1
            w += delta_w
    else:
        while w < np.pi - delta_w/2:
            old_ind = ind
            ind = _point_on_Nsphere_loop(points, delta_w, currDim+1, NDim,
                                         ind, base*np.sin(w), np.sin(w), half)
            points[old_ind:ind, currDim] = base * np.cos(w)
            w += delta_w
    return ind


def _volume_Nball(n):
    """Volume of a unitary N-Ball."""
    if n == 0:
        return 1
    return _surface_Nsphere(n-1) / n


def _surface_Nsphere(n):
    """Surface of a unitary N-Sphere."""
    if n == 0:
        return 2
    return 2 * np.pi * _volume_Nball(n-1)


def get_random_rotation(rotation_axis=None):
    """
    Generate a random rotation matrix.

    :param rotation_axis: The rotation axis.
        If it's 'x', 'y', or 'z', then the rotation is around that axis.
        Otherwise the rotation is totally random.
    :return: A rotation matrix
    """
    if rotation_axis is not None and rotation_axis.lower() in ('x', 'y', 'z'):
        u = np.random.rand() * 2 * np.pi  # random angle between [0, 2pi]
        return convert.angle_axis_to_rot3d(rotation_axis, u)
    else:
        return special_ortho_group.rvs(3)


def get_random_quat(num_pts):
    """
    Get num_pts of unit quaternions on the 4 sphere with a uniform random distribution.

    :param num_pts: The number of quaternions to return
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    u = np.random.rand(3, num_pts)
    u1, u2, u3 = [u[x] for x in range(3)]

    quat = np.zeros((4, num_pts))
    quat[0] = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    quat[1] = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    quat[2] = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    quat[3] = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.transpose(quat)


def get_uniform_quat(num_pts, avoid_symmetric=False):
    """
    Get num_pts of unit quaternions evenly distributed on the 3-sphere.

    :param num_pts: The number of quaternions to return
    :param avoid_symmetric:
        If specified, count opposite quaternions as identical.
    :return: Quaternion list of shape [number of quaternion, 4]
    """
    return points_on_Nsphere(num_pts, 3, half=avoid_symmetric)


def quaternion_product(q1, q0):
    """
    Compute quaternion product, q1 x q0, according to: 
    https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html.
    
    :param q0: first quaternion to rotate by
    :param q1: second quaternion to rotate by
    :return q2: quaternion product q1 x q0, rotation by q0 followed by q1
    """
    p0, p1, p2, p3 = q1
    r0, r1, r2, r3 = q0
    return np.array([r0 * p0 - r1 * p1 - r2 * p2 - r3 * p3,
                     r0 * p1 + r1 * p0 - r2 * p3 + r3 * p2,
                     r0 * p2 + r1 * p3 + r2 * p0 - r3 * p1,
                     r0 * p3 - r1 * p2 + r2 * p1 + r3 * p0])


def get_preferred_orientation_quat(sigma, num_pts, base_quat=None):
    """
    Sample quaternions distributed around a given or random position in a restricted 
    range in SO(3), where the spread of the distribution is determined by sigma.
    
    :param sigma: standard deviation in radians from which to sample angles
    :param num_pts: number of quaternions to generate
    :param base_quat: quaternion about which to distribute samples, random if None
    :return quaternions: array of quaternions with preferred orientations
    """
    quaternions = np.zeros((num_pts,4))
    if base_quat is None:
        base_quat = get_random_quat(1)[0]
    
    for i in range(num_pts):
        R_random = get_random_rotation()
        rot_axis = R_random.dot(np.array([0,0,1]))
        theta = sigma * np.random.randn()
        R_preferred = convert.angle_axis_to_rot3d(rot_axis, theta)
        pref_quat = convert.rotmat_to_quaternion(R_preferred)
        quaternions[i] = quaternion_product(pref_quat, base_quat)
        
    return quaternions


def visualize_quaternions(quaternions):
    """
    Visualize how given quaternions rotate the [0,0,1] unit vector on the surface
    of the unit sphere.
    
    :param quaternions: array of quaternions
    """
    from matplotlib import pyplot as plt
    
    # apply quaternions to [0,0,1] and convert to Cartesian coordinates
    coords = np.zeros((quaternions.shape[0], 3))
    for i in range(coords.shape[0]):
        theta, axis = convert.quaternion_to_angle_axis(quaternions[i])
        R = convert.angle_axis_to_rot3d(axis, theta)
        coords[i] = R.dot(np.array([0,0,1]))
    
    # generate latitutde/longitude lines along unit sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    # plot coordinates and unit sphere mesh
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.2)
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=10, c='r', zorder=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.tight_layout()
    plt.show()
    
    return
