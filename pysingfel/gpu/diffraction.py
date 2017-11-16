from numba import cuda, float64, int32, complex128
from pysingfel.diffraction import *
import math
import numpy as np

###########################################################################
## The following funcitons focus on numba.cuda calculation
###########################################################################

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64, int64[:], int64)')
def calculate_pattern_gpu_back_engine(formFactor, voxelPosition, atomPosition, patternCos, 
                           patternSin, atomTypeNum, splitIndex, voxelNum):    
    row = cuda.grid(1)
    for atom_type in range(atomTypeNum):
        form_factor = formFactor[atom_type,row]
        for atom_iter in range(splitIndex[atom_type], splitIndex[atom_type + 1]):
            if row < voxelNum :
                holder = 0
                for l in range(3):
                    holder += voxelPosition[row, l] * atomPosition[atom_iter , l]
                patternCos[row] +=  form_factor* math.cos(holder)
                patternSin[row] +=  form_factor* math.sin(holder)

def calculate_diffraction_pattern_gpu(reciprocal_space, particle, return_type = 'intensity'):
    """This function can be used to calculate the diffraction field for 
    arbitrary reciprocal space """
    
    # convert the reciprocal space into a 1d series.
    shape = reciprocal_space.shape
    pixel_number = np.prod(shape[:-1])
    reciprocal_space_1d = np.reshape(reciprocal_space,[pixel_number, 3])
    reciprocal_norm_1d = np.sqrt(np.sum(np.square(reciprocal_space_1d),axis=-1))*( 1e-10/2. )
    
    # Calculate atom form factor for the reciprocal space
    form_factor = calculate_atomicFactor(particle, reciprocal_norm_1d, pixel_number)
    
    # Get atom position
    atom_position =  particle.atomPos[:]
    atom_type_num = len(particle.SplitIdx)-1
    
    # create 
    pattern_cos = np.zeros(pixel_number, dtype= np.float64)
    pattern_sin = np.zeros(pixel_number, dtype = np.float64)

    #atom_number = atom_position.shape[0]
    split_index = np.array(particle.SplitIdx)

    cuda_split_index = cuda.to_device(split_index)
    cuda_atom_position = cuda.to_device(atom_position)
    cuda_reciprocal_position = cuda.to_device(reciprocal_space_1d)
    cuda_pattern_cos = cuda.to_device(pattern_cos)
    cuda_pattern_sin = cuda.to_device(pattern_sin)
    cuda_form_factor = cuda.to_device(form_factor)
    
    #Calculate the pattern
    calculate_pattern_gpu_back_engine[(pixel_number + 511)/512, 512](cuda_form_factor,
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
        pattern = np.reshape(np.square(np.abs(pattern_cos+1j*pattern_sin)), shape[:-1])
    elif return_type == "complex_field":
        pattern = np.reshape(pattern_cos+1j*pattern_sin, shape[:-1])
    else:
        print("Set parameter return_type = \"intensity\", program will return "+
              "the intensity of the diffraction field received by the detector. ")
        print("Set parameter return_type = \"complex_field\", program will return "+
              "the complex scattered field received by the detector. ")
    return pattern


###########################################################################
## The following funcitons use tensorflow to do calculation
###########################################################################

'''
import tensorflow as tf

#This method also works. But the efficiency is low. So currently we are using the numba.cuda rather than
#the tensorflow. But if you are interested, you can also check this method.

def calculate_diffraction_volume_gpu(particle, q_space, q_position):
    f_hkl = calculate_atomicFactor_3d(particle, q_space)
    split_index = particle.SplitIdx[:]
    atomPos = particle.atomPos[:]
    size_x, size_y, size_z = q_space.shape
    pattern = np.zeros_like(q_space)
    
    atomNumber = split_index[-1]
    
    atomSpecies = np.zeros(split_index[-1], dtype=int)
    
    for i in range(len(split_index)-1):
        atomSpecies[split_index[i]:split_index[i+1]] = i
        
    form_factor = tf.constant(value= f_hkl, dtype= tf.complex64)
    split_index = tf.constant(value= split_index)
    atom_position = tf.constant(value= atomPos, dtype= tf.complex64)
    reciprocal_position = tf.constant(value= q_position, dtype= tf.complex64)
    atom_species = tf.constant(value = atomSpecies)
    atom_number = tf.constant(value = atomNumber)
        
    pattern_3d =  tf.Variable(tf.zeros([size_x, size_y, size_z], dtype=tf.complex64),
                                  dtype= tf.complex64)
        
    number = tf.Variable(0 ,dtype= tf.int64)
        
    def condition(number, pattern_3d):
        return number < 10
    def body(number, pattern_3d):
        pattern_3d = tf.add(pattern_3d , tf.multiply(form_factor[:,:,:,atom_species[number]],
                                     tf.exp(1j* 
                                           tf.einsum('ijkl,l->ijk', 
                                                     reciprocal_position, atom_position[number]))))
        number += 1
        return number , pattern_3d
        
    init_op = tf.global_variables_initializer()
    index,pattern_holder = tf.while_loop(condition, body, [number , pattern_3d], 
                              shape_invariants=[number.get_shape(), pattern_3d.get_shape()],
                                        parallel_iterations=1000)
        
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      # Run the init operation.
        sess.run(init_op)
        number_of_iteration, pattern = sess.run([index, pattern_holder ])

    return number_of_iteration, np.abs(pattern) ** 2
'''