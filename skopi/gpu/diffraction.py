import skopi.diffraction as pd
import math
import numpy as np

from skopi.util import xp, asnumpy


def calculate_pattern_gpu(form_factor, pixel_position, atom_position, pattern_cos,
                          pattern_sin, atom_type_num, split_index):
    """
    Calculate the scattering field with the provided information.

    :param form_factor: The form factor for each atom.
    :param pixel_position: The position of each pixel in q-space, where q=2*pi*s
    :param atom_position: The position of each atom
    :param pattern_cos: Holder for the real part of the scattering field
    :param pattern_sin: Holder for the imaginary part of the scattering field.
    :param atom_type_num: The number of atom types involved in this particle.
    :param split_index: The ends for each kinds of the atoms
    :return: None
    """
    pattern_cos_x = xp.asarray(pattern_cos)
    pattern_sin_x = xp.asarray(pattern_sin)
    holders = xp.zeros(pattern_cos.size)
    atom_position_x = xp.asarray(atom_position)
    pixel_position_x = xp.asarray(pixel_position)
    form_factor_x = xp.asarray(form_factor)
    
    for atom_type in range(atom_type_num):
        for atom_iter in range(split_index[atom_type], split_index[atom_type + 1]):
            holders[:] = xp.sum(pixel_position_x*atom_position_x[atom_iter,:], axis=1)
            pattern_cos_x += form_factor_x[atom_type, :]  * xp.cos(holders)
            pattern_sin_x += form_factor_x[atom_type, :]  * xp.sin(holders)
    pattern_cos[:] = asnumpy(pattern_cos_x)
    pattern_sin[:] = asnumpy(pattern_sin_x)


def calculate_solvent_pattern_gpu(pixel_position, water_position, pattern_cos,
                                  pattern_sin, water_prefactor, water_num):
    """
    Calculate the scattering field for the ordered solvent contribution.

    :param pixel_position: The position of each pixel in q-space, where q=2*pi*s
    :param atom_position: The position of each atom
    :param pattern_cos: Holder for the real part of the scattering field
    :param pattern_sin: Holder for the imaginary part of the scattering field.
    :param water_prefactor: The form factor equivalent for water.
    :param water_num: The number of ordered water molecules
    :return: None
    """
    pattern_cos_x = xp.asarray(pattern_cos)
    pattern_sin_x = xp.asarray(pattern_sin)
    holders = xp.zeros(pattern_cos.size)
    pixel_position_x = xp.asarray(pixel_position)
    water_position_x = xp.asarray(water_position)
    for water_iter in range(water_num):
        holders[:] = xp.sum(pixel_position_x*water_position_x[water_iter,:], axis=1)
        pattern_cos_x += water_prefactor  * xp.cos(holders)
        pattern_sin_x += water_prefactor  * xp.sin(holders)
    pattern_cos[:] = asnumpy(pattern_cos_x)
    pattern_sin[:] = asnumpy(pattern_sin_x)


def calculate_diffraction_pattern_gpu(reciprocal_space, particle, return_type='intensity'):
    """
    Calculate the diffraction field of the specified reciprocal space.

    :param reciprocal_space: The reciprocal space over which to calculate the diffraction field as s-vectors, where q=2*pi*s
    :param particle: The particle object to calculate the diffraction field.
    :param return_type: 'intensity' to return the intensity field. 'complex_field' to return the full diffraction field.
    :return: The diffraction field.
    """
    """This function can be used to calculate the diffraction field for
    arbitrary reciprocal space """
    if xp == np: 
        print(f'Warning: using numpy to generate diffraction patterns could be slow (set USE_CUPY=1 to use gpu).')
    # convert the reciprocal space into a 1d series.
    shape = reciprocal_space.shape
    pixel_number = int(np.prod(shape[:-1]))
    reciprocal_space_1d = np.reshape(reciprocal_space, [pixel_number, 3])
    reciprocal_norm_1d = np.sqrt(np.sum(np.square(reciprocal_space_1d), axis=-1))
    qvectors_1d = 2*np.pi*reciprocal_space_1d

    # Calculate atom form factor for the reciprocal space, passing in sin(theta)/lambda in per Angstrom
    form_factor = pd.calculate_atomic_factor(particle=particle,
                                             q_space=reciprocal_norm_1d * (1e-10 / 2.),  # For unit compatibility
                                             pixel_num=pixel_number)

    # Get atom position
    atom_position = np.ascontiguousarray(particle.atom_pos[:])
    atom_type_num = len(particle.split_idx) - 1

    # create
    pattern_cos = np.zeros(pixel_number, dtype=np.float64)
    pattern_sin = np.zeros(pixel_number, dtype=np.float64)

    # atom_number = atom_position.shape[0]
    split_index = np.array(particle.split_idx)

    calculate_pattern_gpu(
        form_factor, qvectors_1d, atom_position,
        pattern_cos, pattern_sin, atom_type_num, split_index)

    # Add the hydration layer
    if particle.mesh is not None:
        water_position = np.ascontiguousarray(particle.mesh[particle.solvent_mask,:])
        water_num = np.sum(particle.solvent_mask)
        water_prefactor = particle.solvent_mean_electron_density * particle.mesh_voxel_size**3
        
        calculate_solvent_pattern_gpu(
            qvectors_1d, water_position,
            pattern_cos, pattern_sin, water_prefactor, water_num)

        # Add another contribution if defined, e.g. virus void...
        if particle.other_mask is not None:
            other_position = np.ascontiguousarray(particle.mesh[particle.other_mask,:])
            other_num = np.sum(particle.other_mask)
            other_prefactor = particle.other_mean_electron_density * particle.mesh_voxel_size**3

            calculate_solvent_pattern_gpu(
                qvectors_1d, other_position,
                pattern_cos, pattern_sin, other_prefactor, other_num)

    if return_type == "intensity":
        pattern = np.reshape(np.square(np.abs(pattern_cos + 1j * pattern_sin)), shape[:-1])
        return np.asarray(pattern)
    elif return_type == "complex_field":
        pattern = np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
        return np.asarray(pattern)
    else:
        print("Please set the parameter return_type = 'intensity' or 'complex_field'")
        print("This time, this program return the complex field.")
        pattern = np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
        return np.asarray(pattern)
