#!/usr/bin/env python
# coding: utf-8

########## Holography Imgaging ###############
# In this notebook, we demonstrate how to simulate a holography experiment, where the sample particle is placed near center of the beam (default: at the center of the beam), and the position of the reference particle can be at any user-defined arbitrary postiion.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pysingfel as ps

# Input files
input_dir='../input'
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile=input_dir+'/pdb/3iyf.pdb'
pdbfile2=input_dir+'/pdb/3j03.pdb'

# Load beam
beam = ps.Beam(beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))

# Load and initialize the detector
det = ps.PnccdDetector(geom=geom, beam=beam)
increase_factor = 0.5
print('BEFORE: detector distance = {} m'.format(np.abs(det.distance)))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*np.abs(det.distance)
print('AFTER : detector distance = {} m'.format(det.distance))
#det.distance = 0.25 # reset detector distance for desired resolution

# Create particle object(s)
particle = ps.Particle()
particle.read_pdb(pdbfile, ff='WK')
particle_2 = ps.Particle()
particle_2.read_pdb(pdbfile2, ff='WK')

# Perform Holography experiment
experiment = ps.HOLOExperiment(det, beam, [particle], [particle_2])

# Visaulization
viz = ps.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
img = experiment.generate_image()
viz.imshow(img)
plt.show()

