from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from pysingfel import *
import pysingfel as ps
from pysingfel.util import asnumpy, xp
from pysingfel.build_autoranging_frames import BuildAutoRangeFrames

pwd = os.path.dirname(__file__)

# create particle object(s)
particle = ps.Particle()
particle.read_pdb(os.path.join(pwd,'../input/pdb/3iyf.pdb'), ff='WK')

# load beam
beam = ps.Beam(os.path.join(pwd,'../input/beam/amo86615.beam'))
#beam._n_phot = 1e14 # detector normal
beam._n_phot = 1e17 # detector saturates
#beam._n_phot = 1e20 # detector gets fried

geom = os.path.join(pwd,'../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data')

# load and initialize the detector
det = ps.Epix10kDetector(geom=geom, run_num=0, beam=beam, cameraConfig='fixedMedium')
# reset detector distance for desired resolution
det.distance = 0.25

experiment = ps.SPIExperiment(det, beam, particle)
dp_photons = experiment.generate_image_stack() # generate diffraction field

viz = ps.Visualizer(experiment, diffraction_rings="auto", log_scale=True)

tau = beam.get_photon_energy()/1000.
dp_keV = dp_photons * tau # convert photons to keV

I0width = 0.03
I0min = 0
I0max = 150000
bauf = BuildAutoRangeFrames(det, I0width, I0min, I0max, dp_keV)
bauf.makeFrame()
calib_photons = bauf.frame / tau # convert keV to photons

fig = plt.figure()
img = experiment.det.assemble_image_stack(calib_photons)
viz.imshow(img)
plt.show()
