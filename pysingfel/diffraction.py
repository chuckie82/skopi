import numpy as np
from numba import jit
from scipy.interpolate import CubicSpline
from pysingfel.geometry import reshape_pixels_position_arrays_to_1d
from pysingfel.util import xp


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
    :return:
    """

    half_q = reshape_pixels_position_arrays_to_1d(detector.pixel_distance_reciprocal * 1e-10 / 2.)

    cs = CubicSpline(particle.compton_q_sample, particle.sBound)
    s_bound = cs(half_q)
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
    if xp is not np:
        q_space = q_space.get()
        # CubicSpline are not compatible with Cupy

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


@jit
def get_phase(atom_pos, q_xyz):
    """
    Calculate the phase of the diffraction field due to the specific atom
    :param atom_pos: The atom position
    :param q_xyz: The reciprocal space to calculate.
    :return:
    """
    phase = 2 * np.pi * (atom_pos[0] * q_xyz[:, 0] +
                         atom_pos[1] * q_xyz[:, 1] +
                         atom_pos[2] * q_xyz[:, 2])
    return np.exp(1j * phase)


@jit
def cal(f_hkl, atom_pos, q_xyz, xyz_ind, pixel_number):
    """
    Calculate the diffraction intensity field.

    :param f_hkl: The form factor array
    :param atom_pos:  The atom position array
    :param q_xyz: The reciprocal space to calculate.
    :param xyz_ind: The split index.
    :param pixel_number: number of pixels.
    :return:
    """
    f = np.zeros(pixel_number, dtype=np.complex128)
    for atm in range(atom_pos.shape[0]):
        f += get_phase(atom_pos[atm, :], q_xyz) * f_hkl[xyz_ind[atm], :]
    return np.abs(f) ** 2


def calculate_molecular_form_factor_square(particle, q_space, q_position):
    """
    Calculate the diffraction intensity field of the molecule.

    :param particle: The particle object.
    :param q_space: The reciprocal distance of the pixels.
    :param q_position: The reciprocal position of the pixels.
    :return:
    """
    shape = q_position.shape
    pixel_number = np.prod(shape[:-1])
    q_space_1d = np.reshape(q_space, [pixel_number, ])
    q_position_1d = np.reshape(q_position, [pixel_number, 3])

    f_hkl = calculate_atomic_factor(particle, q_space_1d, pixel_number)
    split_index = particle.split_idx[:]
    xyz_ind = np.zeros(split_index[-1], dtype=int)
    for i in range(len(split_index) - 1):
        xyz_ind[split_index[i]:split_index[i + 1]] = i

    pattern_1d = cal(f_hkl, particle.atom_pos, q_position_1d, xyz_ind, pixel_number)
    pattern = np.reshape(pattern_1d, shape[:-1])

    return pattern
