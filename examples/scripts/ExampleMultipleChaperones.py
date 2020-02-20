import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py as h5
import time, os

import pysingfel as ps
from pysingfel.particlePlacement import position_in_3d

numOpen = 1
numClosed = 1
pwd = os.path.dirname(__file__)

# Create a particle object
particleOp = ps.Particle()
particleOp.read_pdb(os.path.join(pwd,'../input/pdb/3iyf.pdb'), ff='WK')


particleCl = ps.Particle()
particleCl.read_pdb(os.path.join(pwd,'../input/pdb/3j03.pdb'), ff='WK')

# Load beam
beam = ps.Beam(os.path.join(pwd,'../input/beam/amo86615.beam'))

geom = os.path.join(pwd,'../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data')

# Load and initialize the detector
det = ps.PnccdDetector(geom=geom, beam = beam)

tic = time.time()
patternOp = det.get_photons(device='gpu', particle=particleOp)
toc = time.time()
print("It took {:.2f} seconds to finish SPI calculation.".format(toc-tic))

patternCl = det.get_photons(device='gpu', particle=particleCl)

# Calculates 1 diffraction pattern from 1 open chaperones + 1 closed chaperones
pattern = det.get_fxs_photons_slices(device='gpu', beam_focus_radius=beam.focus_xFWHM/2, jet_radius=1e-4, mesh_length=151, particles={particleOp:numOpen,particleCl:numClosed})

pattern_no_polarization = det.remove_polarization(pattern, res=None)
np_img = det.assemble_image_stack(pattern_no_polarization)
mask = np.ones_like(pattern_no_polarization)
np_mask = det.assemble_image_stack(mask)

# write data to HDF5
#with h5.File('mixed_chaperones_and_mask.hdf5', 'w') as f:
with h5.File('test.hdf5', 'w') as f:
    f.create_dataset('img', data=np_img)
    f.create_dataset('mask', data=np_mask, dtype=np.int16)

fig = plt.figure(figsize=(10, 8))
plt.subplot(131)
plt.imshow(det.assemble_image_stack(patternOp),vmin=0, vmax=10)
plt.title('Open state')
plt.subplot(132)
plt.imshow(det.assemble_image_stack(patternCl),vmin=0, vmax=10)
plt.title('Closed state')
plt.subplot(133)
plt.imshow(det.assemble_image_stack(pattern),vmin=0, vmax=10)
plt.title('%i Open + %i Closed'%(numOpen,numClosed))
plt.show()

fig = plt.figure()
polarization = det.polarization_correction
plt.imshow(det.assemble_image_stack(polarization),vmin=0, vmax=1)
plt.colorbar()
plt.title('Polarization')
plt.show()
