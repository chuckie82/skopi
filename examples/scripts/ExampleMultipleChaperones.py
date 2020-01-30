import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pysingfel as ps
import time
import collections


# Create a particle object
particleOp = ps.Particle()
particleOp.read_pdb('../input/3iyf.pdb', ff='WK')


particleCl = ps.Particle()
particleCl.read_pdb('../input/3j03.pdb', ff='WK')

# Load beam
beam = ps.Beam('../input/exp_chuck.beam')

# Load and initialize the detector
det = ps.PnccdDetector(geom = '../input/geometry/0-end.data',
                       beam = beam)

tic = time.time()
patternOp = det.get_photons(device='gpu', particle=particleOp)
toc = time.time()

patternCl = det.get_photons(device='gpu', particle=particleCl)
print("It takes {:.2f} seconds to finish the calculation.".format(toc-tic))

# Calculates 1 diffraction pattern from 1 open chaperones + 1 closed chaperones
pattern = det.get_fxs_photons(device='gpu', beam_focus_radius=beam.focus_xFWHM/2, jet_radius=1e-4, particles={particleOp:5,particleCl:1})

fig = plt.figure(figsize=(10, 8))
plt.subplot(131)
plt.imshow(det.assemble_image_stack(patternOp),vmin=0, vmax=10)
plt.title('Open state')
plt.subplot(132)
plt.imshow(det.assemble_image_stack(patternCl),vmin=0, vmax=10)
plt.title('Closed state')
plt.subplot(133)
plt.imshow(det.assemble_image_stack(pattern),vmin=0, vmax=10)
plt.title('Open+Closed')
plt.show()
