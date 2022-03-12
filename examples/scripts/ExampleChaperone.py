import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import skopi as sk
from skopi.detector.pnccd import PnccdDetector
import time, os

# Create a particle object
script_dir = os.path.dirname(os.path.abspath(__file__))
pdb_dir    = '../input/pdb/3iyf.pdb'
pdb_path   = os.path.join(script_dir, pdb_dir)
particleOp = sk.Particle()
particleOp.read_pdb(pdb_path, ff='WK')
#particleOp.rotate_randomly()

#exit()

pdb_dir    = '../input/pdb/3j03.pdb'
pdb_path   = os.path.join(script_dir, pdb_dir)
particleCl = sk.Particle()
particleCl.read_pdb(pdb_path, ff='WK')

# Load beam
beam_file = '../input/beam/amo86615.beam'
beam_path = os.path.join(script_dir, beam_file)
beam      = sk.Beam(beam_path) 

# Load and initialize the detector
geom_file = '../input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data'
geom_path = os.path.join(script_dir, geom_file)
det       = PnccdDetector(geom = geom_path, beam = beam)

tic = time.time()
patternOp = det.get_photons(device='gpu', particle=particleOp)
toc = time.time()

patternCl = det.get_photons(device='gpu', particle=particleCl)
print("It takes {:.2f} seconds to finish the calculation.".format(toc-tic))

fig = plt.figure(figsize=(10, 8))
plt.subplot(121)
plt.imshow(det.assemble_image_stack(patternOp),vmin=0, vmax=10)
plt.title('Open state')
plt.subplot(122)
plt.imshow(det.assemble_image_stack(patternCl),vmin=0, vmax=10)
plt.title('Closed state')
plt.show()
