import numpy as np
from numba import cuda, float64, int32
from pysingfel.geometry import *

@cuda.jit('void(int64[:,:], float64[:,:], float64[:], float64[:], int64)')
def take_one_slice_backengine(index, weight, diffraction_volume, result, pixel_num):
    pixel = cuda.grid(1)
    if pixel < pixel_num :
        for l in range(8):
            result[pixel] += weight[pixel,l]*diffraction_volume[index[pixel, l]]

def take_n_slice(slice_num):
    pass
    
def take_one_slice(slice_num):
    pass
    