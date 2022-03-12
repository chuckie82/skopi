#!/usr/bin/env python
# coding: utf-8

########## Particle Aggregation ###############
# In this example script, we demonstrate particle sticking/aggregation by introducing the variable sticking into FXS experiment class. When sticking is turned on, i.e. sticking=True, particles stick together to form a single cluster.

import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import skopi as sk
from skopi.particlePlacement import *
from skopi.detector.pnccd import PnccdDetector
import time, os

def drawSphere(xCenter, yCenter, zCenter, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

# Parameter(s)
num = 5

# Input files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile=input_dir+'/pdb/3iyf.pdb'
#pdbfile=input_dir+'/pdb/1uf2.pdb' # Note: Virus particles tend to stick together, but it takes much longer to simulate compared to proteins.

# Load beam
beam = sk.Beam(beamfile)
increase_factor = 1e2
print('BEFORE: # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
print('>>> Increasing the number of photons per pulse by a factor {}'.format(increase_factor))
beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
print('AFTER : # of photons per pulse = {}'.format(beam.get_photons_per_pulse()))
beam.photon_energy = 1600.0 # reset the photon energy
print('photon energy = {} eV'.format(beam.photon_energy))
print('beam radius = {}'.format(beam._focus_xFWHM/2))
print('focus area = {}'.format(beam._focus_area))

# Load and initialize the detector
det = PnccdDetector(geom=geom, beam=beam)
increase_factor = 0.5
print('BEFORE: detector distance = {} m'.format(det.distance))
print('>>> Increasing the detector distance by a factor of {}'.format(increase_factor))
det.distance = increase_factor*det.distance
print('AFTER : detector distance = {} m'.format(det.distance))

# Create particle object(s)
particle = sk.Particle()
particle.read_pdb(pdbfile, ff='WK')

# Perform SPI experiment calculation with particles sticking together to form a single cluster.
# If a multi-particle shot of dispersed rather than aggregated particles is desired, the FXSExperiment 
# should be used instead.
tic = time.time()
experiment = sk.SPIExperiment(det=det, beam=beam, particle=particle, n_part_per_shot=num)
img = experiment.generate_image()
toc = time.time()
print(">>> It took {:.2f} seconds to finish FXS calculation.".format(toc-tic))
viz = sk.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
viz.imshow(img)
plt.show()

# Visualize particle sticking
particle_groups = experiment.generate_new_sample_state()
part_positions = np.array([i[0] for i in particle_groups])
radius = max_radius({particle: num})

x,y,z = part_positions[0][:,0], part_positions[0][:,1], part_positions[0][:,2]
r = np.ones(num)*radius

fig = plt.figure()
ax = fig.gca(projection='3d')
try: 
    ax.set_aspect('equal') # not implemented for all matplotlib versions
except:
    ax.set_aspect('auto')

for (xi,yi,zi,ri) in zip(x,y,z,r):
    (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    ax.plot_wireframe(xs, ys, zs)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='z', nbins=3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
plt.show()
