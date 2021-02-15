import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import skopi as sk
import time, os

# Create a particle object
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
particleOp = sk.Particle()
particleOp.read_pdb(input_dir+'/3iyf.pdb', ff='WK')
#particleOp.rotate_randomly()

#exit()

particleCl = sk.Particle()
particleCl.read_pdb(input_dir+'/3j03.pdb', ff='WK')

# Load beam
beam = sk.Beam(input_dir+'/beam/amo86615.beam') 

# Load and initialize the detector
det = sk.PnccdDetector(geom = input_dir+'/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data', 
                       beam = beam)

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
