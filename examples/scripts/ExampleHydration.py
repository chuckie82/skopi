#!/usr/bin/env python
# coding: utf-8

########## Hydration Layer ###############
# In this notebook, we demonstrate the water surrounding effect by varying the thickness of the hydration layers with the orientation of the particle fixed.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
import time, os
import skopi as sk

# Input files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile=input_dir+'/pdb/3iyf.pdb'

# Load beam
beam = sk.Beam(beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('photon energy = {} eV'.format(beam.photon_energy))

# Load and initialize the detector
det = sk.PnccdDetector(geom=geom, beam=beam)
increase_factor = 0.5
print('BEFORE: detector distance = {} m'.format(det.distance))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*det.distance
print('AFTER : detector distance = {} m'.format(det.distance))

# Create particle object(s)
particle = sk.Particle()
particle.read_pdb(pdbfile, ff='WK')
print('Number of atoms in particle: {}'.format(particle.get_num_atoms()))

# Generate SPI diffraction patterns for various thickness of the hydration layers with the orientation of the particle fixed.
imgs = dict()
orientation = np.array([[1., 1., 0., 0.]])/np.sqrt(2)
thickness = np.arange(0,15,5)
for i in range(len(thickness)):
    hydration_layer_thickness = thickness[i]*1e-10
    mesh_voxel_size = 2.0*1e-10
    particle.set_hydration_layer_thickness(hydration_layer_thickness)
    particle.create_masks()
    experiment = sk.SPIExperiment(det=det, beam=beam, particle=particle, jet_radius=1e-4, n_part_per_shot=1)
    experiment.set_orientations(orientation)
    imgs[i] = experiment.generate_image()

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,12))
im1 = ax1.imshow(imgs[0],norm=LogNorm())
ax1.set_title(r'Hydration layer = {} $\rm \AA$'.format(thickness[0]))
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
im2 = ax2.imshow(imgs[1],norm=LogNorm())
ax2.set_title(r'Hydration layer = {} $\rm \AA$'.format(thickness[1]))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
im3 = ax3.imshow(imgs[2],norm=LogNorm())
ax3.set_title(r'Hydration layer = {} $\rm \AA$'.format(thickness[2]))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)
