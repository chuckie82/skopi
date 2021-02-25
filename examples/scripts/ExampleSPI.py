#!/usr/bin/env python
# coding: utf-8

########## SPI Experiment ###############
# In this notebook, we demonstrate how to simulate an SPI experiment, where a diffraction volume of the particle is computed in the reciprocal space, and the diffraction patterns are sliced from the diffraction volume in random orientations.
# Input parameters including (1) beam, (2) detector, (3) gas jet radius, (4) particle(s), (5) number of particle per shot, (6) sticking=Ture or False are needed for the SPI Experiment class.
# sticking=True is used to simulate multiple-particle hit (which we must veto before reconstruction), where particles are forced to form a single cluster.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import time, os
import skopi as sk

# Parameter(s)
num = 1

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

# Perform SPI calculation
tic = time.time()
experiment = sk.SPIExperiment(det=det, beam=beam, jet_radius=1e-4, particles=[particle], n_part_per_shot=num, sticking=False)
img = experiment.generate_image()
toc = time.time()
print(">>> It took {:.2f} seconds to finish SPI calculation.".format(toc-tic))
viz = sk.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
viz.imshow(img)
plt.show()
