#!/usr/bin/env python
# coding: utf-8

########## SPI Experiment ###############
# We demonstrate how to simulate an SPI experiment, where a diffraction volume of the particle is computed in the reciprocal space, and the diffraction patterns are sliced from the diffraction volume in random orientations.
# Input parameters for SPIExperiment class including (1) detector, (2) beam, and (3) particle.

import numpy as np
import matplotlib
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
increase_factor = 0.3
print('BEFORE: detector distance = {} m'.format(det.distance))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*det.distance
print('AFTER : detector distance = {} m'.format(det.distance))

# Create particle object(s)
particle = sk.Particle()
particle.read_pdb(pdbfile, ff='WK')

# Perform SPI calculation
tic = time.time()
experiment = sk.SPIExperiment(det=det, beam=beam, particle=particle, n_part_per_shot=num) 
img = experiment.generate_image()
toc = time.time()
print(">>> It took {:.2f} seconds to finish SPI calculation.".format(toc-tic))
viz = sk.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
viz.imshow(img)
plt.show()
