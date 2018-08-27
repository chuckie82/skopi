from numba import cuda, float64, int32, complex128
from pysingfel.diffraction import *
import math
import numpy as np


###########################################################################
# The following funcitons focus on numba.cuda calculation
###########################################################################

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64, int64[:], int64)')
def calculate_pattern_gpu_back_engine(formFactor, voxelPosition, atomPosition, patternCos,
                                      patternSin, atomTypeNum, splitIndex, voxelNum):
    row = cuda.grid(1)
    for atom_type in range(atomTypeNum):
        form_factor = formFactor[atom_type, row]
        for atom_iter in range(splitIndex[atom_type], splitIndex[atom_type + 1]):
            if row < voxelNum:
                holder = 0
                for l in range(3):
                    holder += voxelPosition[row, l] * atomPosition[atom_iter, l]
                patternCos[row] += form_factor * math.cos(holder)
                patternSin[row] += form_factor * math.sin(holder)


def calculate_diffraction_pattern_gpu(reciprocal_space, particle, return_type='intensity'):
    """This function can be used to calculate the diffraction field for 
    arbitrary reciprocal space """

    # convert the reciprocal space into a 1d series.
    shape = reciprocal_space.shape
    pixel_number = np.prod(shape[:-1])
    reciprocal_space_1d = np.reshape(reciprocal_space, [pixel_number, 3])
    reciprocal_norm_1d = np.sqrt(np.sum(np.square(reciprocal_space_1d), axis=-1)) * (1e-10 / 2.)

    # Calculate atom form factor for the reciprocal space
    form_factor = calculate_atomicFactor(particle, reciprocal_norm_1d, pixel_number)

    # Get atom position
    atom_position = particle.atomPos[:]
    atom_type_num = len(particle.SplitIdx) - 1

    # create 
    pattern_cos = np.zeros(pixel_number, dtype=np.float64)
    pattern_sin = np.zeros(pixel_number, dtype=np.float64)

    # atom_number = atom_position.shape[0]
    split_index = np.array(particle.SplitIdx)

    cuda_split_index = cuda.to_device(split_index)
    cuda_atom_position = cuda.to_device(atom_position)
    cuda_reciprocal_position = cuda.to_device(reciprocal_space_1d)
    cuda_pattern_cos = cuda.to_device(pattern_cos)
    cuda_pattern_sin = cuda.to_device(pattern_sin)
    cuda_form_factor = cuda.to_device(form_factor)

    # Calculate the pattern
    calculate_pattern_gpu_back_engine[(pixel_number + 511) / 512, 512](cuda_form_factor,
                                                                       cuda_reciprocal_position,
                                                                       cuda_atom_position,
                                                                       cuda_pattern_cos,
                                                                       cuda_pattern_sin,
                                                                       atom_type_num,
                                                                       cuda_split_index,
                                                                       pixel_number)

    cuda_pattern_cos.to_host()
    cuda_pattern_sin.to_host()

    if return_type == "intensity":
        pattern = np.reshape(np.square(np.abs(pattern_cos + 1j * pattern_sin)), shape[:-1])
        return pattern
    elif return_type == "complex_field":
        pattern = np.reshape(pattern_cos + 1j * pattern_sin, shape[:-1])
        return pattern
    else:
        print("Set parameter return_type = \"intensity\", program will return " +
              "the intensity of the diffraction field received by the detector. ")
        print("Set parameter return_type = \"complex_field\", program will return " +
              "the complex scattered field received by the detector. ")
