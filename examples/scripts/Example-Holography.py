import sys
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append('/cds/home/i/iris/pysingfel')
import pysingfel as ps

input_dir='../input'
beamfile=input_dir+'/beam/amo86615.beam'
geom=input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
pdbfile=input_dir+'/pdb/3iyf.pdb'

beam = ps.Beam(beamfile)
beam.photon_energy = 1600.0 # reset the photon energy
print ("photon energy=", beam.photon_energy)
print ("beam radius=", beam._focus_xFWHM/2)
print ("focus area=", beam._focus_area)
print ("number of photons per shot=", beam._n_phot)

det = ps.PnccdDetector(geom=geom, beam=beam)
det.distance = 0.581*0.5
print ("detector distance=", det.distance)

particle = ps.Particle()
particle.read_pdb(pdbfile, ff='WK')

pdbfile2=input_dir+'/pdb/3j03.pdb'
particle_2 = ps.Particle()
particle_2.read_pdb(pdbfile2, ff='WK')

experiment = ps.HOLOExperiment(det, beam, [particle], [particle_2], 3, 3)

viz = ps.Visualizer(experiment, diffraction_rings="auto", log_scale=True)
img = experiment.generate_image()
viz.imshow(img)
plt.show()

