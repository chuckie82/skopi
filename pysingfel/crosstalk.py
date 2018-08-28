import numpy as np
from numba import jit

def make_cross_talk_lib(data_num = 100000, effect_distance=1., boundary = 2, average_adu = 130. 
                        ,path = './cross_talk_lib.npy'):
    
    data_num = int(data_num)
    
    side = int(2*boundary + 1)
    
    data = np.zeros((data_num , 2 + side*side))
    hit_point = np.random.rand(2,data_num) - 0.5

    Xs, Ys = np.meshgrid(np.array(range(-boundary, boundary+1)) , 
                         np.array(range(-boundary, boundary+1)))

    coordinate = np.zeros((side,side,2))
    coordinate[:,:,0] = Xs
    coordinate[:,:,1] = Ys

    coordinate = np.reshape(coordinate, [side*side,2])

    for l in range(data_num):
        distances = np.sum(np.square(coordinate - hit_point[np.newaxis, :, l]), axis=-1)
        data[l,0:2] = hit_point[:,l]
        data[l,2:]  = distances


    #convert to density
    data[:,2:] = np.exp(- data[:,2:]/effect_distance)

    # normalize
    norm = np.sum(data[:,2:],axis= -1)
    data[:,2:] = data[:,2:]/  norm[:,np.newaxis]

    lib = np.zeros((data_num, side*side))
    for l in range(data_num):
        lib[l,:] = np.random.multinomial(average_adu, data[l,2:])
    np.save(path , lib)
    
    print(" The cross talk effect library is saved to" + path)
    
    

@jit
def _cross_talk_effect(library, photons, shape, lib_length, boundary):
    
    # Create the variable to hold the value
    adu = np.zeros((shape[0] ,shape[1] + boundary, shape[2] + boundary))
    
    for l in range(shape[0]):
        for m in range(shape[1]):
            for n in range(shape[2]):
                index = np.random.randint(low = 0, high=lib_length, size=(photons[l,m,n],) )
                adu[l, m:m+5, n:n+5] += np.sum(library[index,:], axis=0)
    return adu

def add_cross_talk_effect_panel(lib_path, photons):
    lib = np.load(lib_path)
    lib_length = lib.shape[0]
    # calculate the boundary of the model
    boundary = np.sqrt(lib.shape[1])
    boundary = boundary - 1
    boundary = int(boundary)
    
    library = np.reshape(lib ,[lib_length, boundary+1, boundary+1])
    # build the adu from the lib
    shape = photons.shape
    adu = _cross_talk_effect(library, photons, shape, lib_length, boundary)
        
    return adu[:, boundary/2:-boundary/2, boundary/2:-boundary/2]
