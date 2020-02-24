import sys
sys.path.append("../..")

import os
import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
import time

import pysingfel as ps
import pysingfel.gpu as pg


# Set default matplotlib parameters
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = None
matplotlib.rcParams['image.cmap'] = 'jet'


# Create a particle object
particle = ps.Particle()
particle.read_pdb('../input/pdb/3iyf.pdb', ff='WK')

# Load beam
beam = ps.Beam('../input/beam/amo86615.beam') 

# Load and initialize the detector
det = ps.PnccdDetector(
    geom='../input/lcls/amo86615/PNCCD::CalibV1/'
         'Camp.0:pnCCD.1/geometry/0-end.data', 
    beam=beam)

mesh_length = 51

mesh, voxel_length = det.get_reciprocal_mesh(voxel_number_1d=mesh_length)

volume = pg.calculate_diffraction_pattern_gpu(
    mesh, particle, return_type='intensity')

pixel_momentum = det.pixel_position_reciprocal

orientation = np.array([1., 0., 0., 0.])

slice_ = ps.geometry.take_slice(
    volume, voxel_length, pixel_momentum, orientation, inverse=True)

img = det.assemble_image_stack(slice_)

plt.imshow(img)
plt.show()
