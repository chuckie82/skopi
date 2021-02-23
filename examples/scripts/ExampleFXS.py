#!/usr/bin/env python
# coding: utf-8

########## FXS Experiment ###############
# In this notebook, we demonstrate how to simulate a FXS experiment by introducing the capability to place multiple particles (with user-defined ratio) at the interaction point. Interference can sometimes be observed in the diffraction pattern.
# Input parameters including (1) beam, (2) detector, (3) particle(s) are needed for the FXS Experiment class.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
import time, os
import skopi as sk

# Parameters
numOp = 1
numCl = 1
num = 2

# Input files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile1=input_dir+'/pdb/3iyf.pdb'
pdbfile2=input_dir+'/pdb/3j03.pdb'

# Load beam
beam = sk.Beam(beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))

# Load and initialize the detector
det = sk.PnccdDetector(geom=geom, beam=beam)
increase_factor = 0.5
print('BEFORE: detector distance = {} m'.format(det.distance))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*det.distance
print('AFTER : detector distance = {} m'.format(det.distance))

# Create particle object(s)
particleOp = sk.Particle()
particleOp.read_pdb(pdbfile1, ff='WK')

particleCl = sk.Particle()
particleCl.read_pdb(pdbfile2, ff='WK')

# Perform FXS experiment with one particle type
tic = time.time()
experiment = sk.FXSExperiment(det=det, beam=beam, jet_radius=1e-4, particles=[particleOp], n_part_per_shot=numOp, sticking=False)
patternOp = experiment.generate_image_stack()
toc = time.time()
print(">>> It took {:.2f} seconds to finish FXS calculation.".format(toc-tic))

# Perform FXS experiment with one particle type
experiment = sk.FXSExperiment(det=det, beam=beam, jet_radius=1e-4, particles=[particleCl], n_part_per_shot=numCl, sticking=False)
patternCl = experiment.generate_image_stack()

# Perform FXS experiment with two particle types
# calculate 1 diffraction pattern from 2 particles, where each particle has 50% chance of being Open or Closed
# (end up in 25% with two Open, 25% with two Closed, and 50% with one of each)
experiment = sk.FXSExperiment(det=det, beam=beam, jet_radius=1e-4, particles=[particleOp, particleCl], n_part_per_shot=num, sticking=False)
pattern = experiment.generate_image_stack()

# Remove polarization
pattern_no_polarization = det.remove_polarization(pattern, res=None)
np_img = det.assemble_image_stack(pattern_no_polarization)
mask = np.ones_like(pattern_no_polarization)
np_mask = det.assemble_image_stack(mask)

# Write data to HDF5
with h5.File('mixed_chaperones_and_mask.hdf5', 'w') as f:
    f.create_dataset('img', data=np_img)
    f.create_dataset('mask', data=np_mask, dtype=np.int16)

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,12))
im1 = ax1.imshow(det.assemble_image_stack(patternOp),norm=LogNorm())
ax1.set_title('Open Chaperone (photons)')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
im2 = ax2.imshow(det.assemble_image_stack(patternCl),norm=LogNorm())
ax2.set_title('Closed Chaperone (photons)')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
im3 = ax3.imshow(det.assemble_image_stack(pattern),norm=LogNorm())
ax3.set_title('Mixed Chaperones (photons)')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,12))
polarization = det.polarization_correction
im1 = ax1.imshow(det.assemble_image_stack(polarization),vmin=0.995, vmax=1)
ax1.set_title('Polarization')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
im2 = ax2.imshow(det.assemble_image_stack(mask),vmin=0, vmax=1)
ax2.set_title('Polarization corrected region')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
plt.show()
