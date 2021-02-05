#!/usr/bin/env python
# coding: utf-8

########## SPI Pattern Generation Under (1) Monochromatic Beam and (2) SASE Operation Mode ###############
# In this notebook, the same particle with the same orientation are picked so that we can compare the difference between the SPI patterns generated under monochromatic beam and SASE operation mode. The SASE blurring effect is expected to be observed.
# To fix the particle orientation, set a random seed in get_random_quat in pysingfel/geometry/generate.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import time, os
import pysingfel as ps

# Input files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile=input_dir+'/pdb/3iyf.pdb'

# Load beam
beam = ps.Beam(beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('BEFORE : photon energy = {} eV'.format(beam.photon_energy))
beam.photon_energy = 7120.0 # reset the photon energy to that in LS49
print('AFTER : photon energy = {} eV'.format(beam.photon_energy))

# Load and initialize the detector
det = ps.PnccdDetector(geom=geom, beam=beam)
increase_factor = 0.5
print('BEFORE: detector distance = {} m'.format(det.distance))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*det.distance
print('AFTER : detector distance = {} m'.format(det.distance))

# Create particle object(s)
particle = ps.Particle()
particle.read_pdb(pdbfile, ff='WK')

# Case1: Monochromatic Beam, SPI
tic = time.time()
experiment = ps.SPIExperiment(det, beam, particle)
img = experiment.generate_image()
toc = time.time()
print(">>> It took {:.2f} seconds to finish Monochromatic Beam, SPI calculation.".format(toc-tic))
viz = ps.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
plt.title('Monochromatic Beam, SPI')
viz.imshow(img)
plt.show()

# Case2: SASE Beam, SPI
Beam = ps.Beam
sase = ps.SASEBeam(mu=7120, sigma=10, n_spikes=100, fname=beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(sase.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
sase.set_photons_per_pulse(increase_factor*sase.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(sase.get_photons_per_pulse()))
print('sase beam mean photon energy = {} eV'.format(sase.photon_energy))
tic = time.time()
experiment = ps.SPIExperiment(det, sase, particle)
img = experiment.generate_image()
toc = time.time()
print(">>> It took {:.2f} seconds to finish SASE Beam, SPI calculation.".format(toc-tic))
viz = ps.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
plt.title('SASE Beam, SPI')
viz.imshow(img)
plt.show()
