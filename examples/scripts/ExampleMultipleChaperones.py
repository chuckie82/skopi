from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import time, os
from pysingfel import *
import pysingfel as ps

numOp = 1
numCl = 1
num = 2
pwd = os.path.dirname(__file__)

# create particle object(s)
particleOp = ps.Particle()
particleOp.read_pdb(os.path.join(pwd,'../input/pdb/3iyf.pdb'), ff='WK')

particleCl = ps.Particle()
particleCl.read_pdb(os.path.join(pwd,'../input/pdb/3j03.pdb'), ff='WK')

# load beam
beam = ps.Beam(os.path.join(pwd,'../input/beam/amo86615.beam'))

geom = os.path.join(pwd,'../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data')

# load and initialize the detector
det = ps.PnccdDetector(geom=geom, beam=beam)

tic = time.time()
experiment = ps.FXSExperiment(det, beam, [particleOp], numOp)
patternOp = experiment.generate_image_stack()
toc = time.time()
print("It took {:.2f} seconds to finish SPI calculation.".format(toc-tic))

experiment = ps.FXSExperiment(det, beam, [particleCl], numCl)
patternCl = experiment.generate_image_stack()

# calculate 1 diffraction pattern from 2 particles, where each particle has 50% chance of being Open or Closed
# (end up in 25% with two Open, 25% with two Closed, and 50% with one of each)
experiment = ps.FXSExperiment(det, beam, [particleOp, particleCl], num)
pattern = experiment.generate_image_stack()

pattern_no_polarization = det.remove_polarization(pattern, res=None)
np_img = det.assemble_image_stack(pattern_no_polarization)
mask = np.ones_like(pattern_no_polarization)
np_mask = det.assemble_image_stack(mask)

# write data to HDF5
with h5.File('mixed_chaperones_and_mask.hdf5', 'w') as f:
    f.create_dataset('img', data=np_img)
    f.create_dataset('mask', data=np_mask, dtype=np.int16)

fig = plt.figure(figsize=(10, 8))
plt.subplot(131)
plt.imshow(det.assemble_image_stack(patternOp),norm=LogNorm())
plt.colorbar()
plt.title('Open')
plt.subplot(132)
plt.imshow(det.assemble_image_stack(patternCl),norm=LogNorm())
plt.colorbar()
plt.title('Closed')
plt.subplot(133)
plt.imshow(det.assemble_image_stack(pattern),norm=LogNorm())
plt.colorbar()
plt.title('Mixed')
plt.show()

fig = plt.figure()
polarization = det.polarization_correction
plt.imshow(det.assemble_image_stack(polarization),vmin=0, vmax=1)
plt.colorbar()
plt.title('Polarization')
plt.show()
