import numpy as np
from scipy.interpolate import CubicSpline
from skopi.util import xp, asnumpy


def calculate_thomson(ang):
    """
    Calculate the Thomson scattering
    :param ang: The angle of the scattered particle.
    :return:
    """
    # Should fix this to accept angles mu and theta
    re = 2.81793870e-15  # classical electron radius (m)
    p = (1 + np.cos(ang)) / 2.
    return re ** 2 * p  # Thomson scattering (m^2)


def calculate_compton(particle, detector):
    """
    Calculate the contribution to the diffraction pattern from compton scattering.

    :param particle: The particle object
    :param detector: The detector object
    :return compton: Compton contribution in shape of detector array
    """

    from scipy.interpolate import InterpolatedUnivariateSpline
    half_q = detector.pixel_distance_reciprocal * 1e-10 / 2.

    # cubic interpolation for pixels in q-range of compton_q_sample
    f = InterpolatedUnivariateSpline(particle.compton_q_sample, particle.sBound, k=3)
    s_bound = f(half_q)
    
    # reduce inteprolation order for pixels that require extrapolation
    f = InterpolatedUnivariateSpline(particle.compton_q_sample, particle.sBound, k=1)
    extrapolate_indices = np.where(half_q>particle.compton_q_sample.max())
    s_bound[extrapolate_indices] = f(half_q[extrapolate_indices])

    if isinstance(particle.nFree, (list, tuple, np.ndarray)):
        # if iterable, take first element to be number of free electrons
        n_free = particle.nFree[0]
    else:
        # otherwise assume to be a single number
        n_free = particle.nFree
    compton = s_bound + n_free
    return compton


def calculate_atomic_factor(particle, q_space, pixel_num):
    """
    Calculate the atomic form factor for each atom at each momentum
    :param particle: The particle object
    :param q_space: The reciprocal to calculate
    :param pixel_num: The number of pixels.
    :return:
    """
    q_space = asnumpy(q_space)  # CubicSpline is not compatible with Cupy

    f_hkl = np.zeros((particle.num_atom_types, pixel_num))
    q_space_1d = np.reshape(q_space, [pixel_num, ])

    if particle.num_atom_types == 1:
        cs = CubicSpline(particle.q_sample, particle.ff_table[:])  # Use cubic spline
        f_hkl[0, :] = cs(q_space_1d)  # interpolate
    else:
        for atm in range(particle.num_atom_types):
            cs = CubicSpline(particle.q_sample, particle.ff_table[atm, :])  # Use cubic spline
            f_hkl[atm, :] = cs(q_space_1d)  # interpolate

    f_hkl = np.reshape(f_hkl, (particle.num_atom_types,) + q_space.shape)
    return xp.asarray(f_hkl)

