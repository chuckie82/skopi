import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py as h5
import time, os

import skopi as sk
from skopi.particlePlacement import position_in_3d

numCyclohexane = 500
numDinitro = 500
numPyrene = 1000
pwd = os.path.dirname(__file__)

# Create a particle object
particleC = sk.Particle()
particleC.read_pdb(os.path.join(pwd,'../input/pdb/cyclohexane.pdb'), ff='WK')

particleD = sk.Particle()
particleD.read_pdb(os.path.join(pwd,'../input/pdb/dinitro.pdb'), ff='WK')

particleP = sk.Particle()
particleP.read_pdb(os.path.join(pwd,'../input/pdb/pyrene.pdb'), ff='WK')

# Load beam
beam = sk.Beam(os.path.join(pwd,'../input/beam/temp.beam'))

geom = os.path.join(pwd,'../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data')

# Load and initialize the detector
det = sk.PnccdDetector(geom=geom, beam = beam)

tic = time.time()
patternC = det.get_photons(device='gpu', particle=particleC)
toc = time.time()
print("It took {:.2f} seconds to finish SPI calculation.".format(toc-tic))

patternD = det.get_photons(device='gpu', particle=particleD)
patternP = det.get_photons(device='gpu', particle=particleP)

# Calculates 1 diffraction pattern from 1 open chaperones + 1 closed chaperones
tic = time.time()
pattern = det.get_fxs_photons_slices(device='gpu', beam_focus_radius=beam.focus_xFWHM/2, jet_radius=1e-5, mesh_length=151, particles={particleC:numCyclohexane,particleD:numDinitro,particleP:numPyrene})
toc = time.time()
print("It took {:.2f} seconds to finish FXS calculation.".format(toc-tic))

fig = plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.imshow(det.assemble_image_stack(patternC),vmin=0)
plt.title('Cyclohexane'); plt.colorbar()
plt.subplot(222)
plt.imshow(det.assemble_image_stack(patternD),vmin=0)
plt.title('Dinitro'); plt.colorbar()
plt.subplot(223)
plt.imshow(det.assemble_image_stack(patternP),vmin=0)
plt.title('Pyrene'); plt.colorbar()
plt.subplot(224)
plt.imshow(det.assemble_image_stack(pattern),vmin=0)
plt.title('%i Cy + %i Di + %i Py'%(numCyclohexane,numDinitro,numPyrene)); plt.colorbar()
plt.show()

