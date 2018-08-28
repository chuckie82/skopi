import numba as nb
from numba import jit
from scipy.interpolate import CubicSpline
import numpy as np


def calculate_thomson(ang):
    # Should fix this to accept angles mu and theta
    re = 2.81793870e-15  # classical electron radius (m)
    P = (1 + np.cos(ang)) / 2.
    return re ** 2 * P  # Thomson scattering (m^2)


def calculate_compton(particle, detector):
    """
    Calculate the contribution to the diffraction pattern from compton scattering.
    """
    half_q = detector.q_mod * 1e-10 / 2.
    cs = CubicSpline(particle.comptonQSample, particle.sBound)
    S_bound = cs(half_q)
    if isinstance(particle.nFree, (list, tuple, np.ndarray)):
        # if iterable, take first element to be number of free electrons
        N_free = particle.nFree[0]
    else:
        # otherwise assume to be a single number
        N_free = particle.nFree
    Compton = S_bound + N_free
    return Compton


def calculate_atomicFactor(particle, q_space, pixel_num):
    f_hkl = np.zeros((particle.numAtomTypes, pixel_num))
    q_space_1d = np.reshape(q_space, [pixel_num, ])

    if particle.numAtomTypes == 1:
        cs = CubicSpline(particle.qSample, particle.ffTable[:])  # Use cubic spline
        f_hkl[0, :] = cs(q_space_1d)  # interpolate
    else:
        for atm in range(particle.numAtomTypes):
            cs = CubicSpline(particle.qSample, particle.ffTable[atm, :])  # Use cubic spline
            f_hkl[atm, :] = cs(q_space_1d)  # interpolate

    return np.reshape(f_hkl, [particle.numAtomTypes, ] + list(q_space.shape))


@jit
def Phase(atomPos, q_xyz):
    phase = 2 * np.pi * (atomPos[0] * q_xyz[:, 0] + atomPos[1] * q_xyz[:, 1] + atomPos[2] * q_xyz[:, 2])
    return np.exp(1j * phase)


@jit
def cal(f_hkl, atomPos, q_xyz, xyzInd, pixel_number):
    F = np.zeros(pixel_number, dtype=nb.c16)
    for atm in range(atomPos.shape[0]):
        F += Phase(atomPos[atm, :], q_xyz) * f_hkl[xyzInd[atm], :]
    return np.abs(F) ** 2


def calculate_molecularFormFactorSq(particle, q_space, q_position):
    shape = q_position.shape
    pixel_number = np.prod(shape[:-1])
    q_space_1d = np.reshape(q_space, [pixel_number, ])
    q_position_1d = np.reshape(q_position, [pixel_number, 3])

    f_hkl = calculate_atomicFactor(particle, q_space_1d, pixel_number)
    split_index = particle.SplitIdx[:]
    xyzInd = np.zeros(split_index[-1], dtype=int)
    for i in range(len(split_index) - 1):
        xyzInd[split_index[i]:split_index[i + 1]] = i

    pattern_1d = cal(f_hkl, particle.atomPos, q_position_1d, xyzInd, pixel_number)
    pattern = np.reshape(pattern_1d, shape[:-1])

    return pattern
