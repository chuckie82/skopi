from numba import cuda, float64, int64, int32, complex128
import pysingfel.diffraction as pd
import math
import numpy as np
import os

xp = np
if os.environ.get('USE_CUPY') != '0':
    try:
        import cupy as xp
    except ImportError:
        pass


@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64, int64[:], int64)')
def calculate_pattern_gpu_back_engine(form_factor, pixel_position, atom_position, pattern_cos,
                                      pattern_sin, atom_type_num, split_index, pixel_num):
    """
    Calculate the scattering field with the provided information.

    :param form_factor: The form factor for each atom.
    :param pixel_position: The position of each pixel.
    :param atom_position: The position of each atom
    :param pattern_cos: Holder for the real part of the scattering field
    :param pattern_sin: Holder for the imaginary part of the scattering field.
    :param atom_type_num: The number of atom types involved in this particle.
    :param split_index: The ends for each kinds of the atoms
    :param pixel_num: The number of pixels to calculate.
    :return: None
    """
    row = cuda.grid(1)
    for atom_type in range(atom_type_num):
        local_form_factor = form_factor[atom_type, row]
        for atom_iter in range(split_index[atom_type], split_index[atom_type + 1]):
            if row < pixel_num:
                holder = 0
                for l in range(3):
                    holder += pixel_position[row, l] * atom_position[atom_iter, l]
                pattern_cos[row] += local_form_factor * math.cos(holder)
                pattern_sin[row] += local_form_factor * math.sin(holder)


def calculate_diffraction_pattern_gpu(reciprocal_space, particle, return_type='intensity'):
    """
    Calculate the diffraction field of the specified reciprocal space.

    :param reciprocal_space: The reciprocal space over which to calculate the diffraction field.
    :param particle: The particle object to calculate the diffraction field.
    :param return_type: 'intensity' to return the intensity field. 'complex_field' to return the full diffraction field.
    :return: The diffraction field.
    """
    """This function can be used to calculate the diffraction field for
    arbitrary reciprocal space """
    # convert the reciprocal space into a 1d series.
    shape = reciprocal_space.shape
    pixel_number = np.prod(shape[:-1])
    reciprocal_space_1d = np.reshape(reciprocal_space, [pixel_number, 3])
    reciprocal_norm_1d = np.sqrt(np.sum(np.square(reciprocal_space_1d), axis=-1))

    # Calculate atom form factor for the reciprocal space
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

    cuda_split_index = cuda.to_device(split_index)
    cuda_atom_position = cuda.to_device(atom_position)
    cuda_reciprocal_position = cuda.to_device(reciprocal_space_1d)
    cuda_form_factor = cuda.to_device(form_factor)

    # Calculate the pattern
    calculate_pattern_gpu_back_engine[(pixel_number + 511) // 512, 512](
        cuda_form_factor, cuda_reciprocal_position, cuda_atom_position,
        pattern_cos, pattern_sin, atom_type_num, cuda_split_index,
        pixel_number)

    if return_type == "intensity":
        pattern = np.reshape(np.square(np.abs(pattern_cos + 1j * pattern_sin)), shape[:-1])
        return pattern
    elif return_type == "complex_field":
        pattern = np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
        return pattern
    else:
        print("Please set the parameter return_type = 'intensity' or 'complex_field'")
        print("This time, this program return the complex field.")
        return np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])


def calculate_fxs_diffraction_pattern_gpu(reciprocal_space, particle, coords, return_type='intensity'):
    """
    Calculate the diffraction field of the specified reciprocal space.

    :param reciprocal_space: The reciprocal space over which to calculate the diffraction field.
    :param particle: The particle object to calculate the diffraction field.
    :param return_type: 'intensity' to return the intensity field. 'complex_field' to return the full diffraction field.
    :return: The diffraction field.
    """
    """This function can be used to calculate the diffraction field for
    arbitrary reciprocal space """
    # convert the reciprocal space into a 1d series.
    shape = reciprocal_space.shape
    pixel_number = np.prod(shape[:-1])
    reciprocal_space_1d = np.reshape(reciprocal_space, [pixel_number, 3])
    reciprocal_norm_1d = np.sqrt(np.sum(np.square(reciprocal_space_1d), axis=-1))

    # Calculate atom form factor for the reciprocal space
    form_factor = pd.calculate_atomic_factor(particle=particle,
                                             q_space=reciprocal_norm_1d * (1e-10 / 2.),  # For unit compatibility
                                             pixel_num=pixel_number)

    # Get atom position
    print ("particle.atom_pos[:].shape", particle.atom_pos[:].shape)
    print ("coords.shape", coords.shape)
    atom_position = np.ascontiguousarray(particle.atom_pos[:]+coords)
    atom_type_num = len(particle.split_idx) - 1

    # create
    pattern_cos = np.zeros(pixel_number, dtype=np.float64)
    pattern_sin = np.zeros(pixel_number, dtype=np.float64)

    # atom_number = atom_position.shape[0]
    split_index = np.array(particle.split_idx)

    cuda_split_index = cuda.to_device(split_index)
    cuda_atom_position = cuda.to_device(atom_position)
    cuda_reciprocal_position = cuda.to_device(reciprocal_space_1d)
    cuda_form_factor = cuda.to_device(form_factor)

    # Calculate the pattern
    calculate_pattern_gpu_back_engine[(pixel_number + 511) // 512, 512](
        cuda_form_factor, cuda_reciprocal_position, cuda_atom_position,
        pattern_cos, pattern_sin, atom_type_num, cuda_split_index,
        pixel_number)

    if return_type == "intensity":
        pattern = np.reshape(np.square(np.abs(pattern_cos + 1j * pattern_sin)), shape[:-1])
        return pattern
    elif return_type == "complex_field":
        pattern = np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
        return pattern
    else:
        print("Please set the parameter return_type = 'intensity' or 'complex_field'")
        print("This time, this program return the complex field.")
        return np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
