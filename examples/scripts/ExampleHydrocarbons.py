import sys
sys.path.append("/reg/neh/home/yoon82/Software/pysingfel/")

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pysingfel as ps
import time

# Create a particle object
particleOp = ps.Particle()
particleOp.read_pdb('../input/pyrene.pdb', ff='WK')

# Load beam
beam = ps.Beam('../input/exp_chuck.beam') 

# Load and initialize the detector
det = ps.PnccdDetector(geom = '../../lcls_detectors/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/hydrocarbon.data', 
                       beam = beam)

tic = time.time()
patternOp = det.get_photons(device='gpu', particle=particleOp)
toc = time.time()
print("It takes {:.2f} seconds to finish the calculation.".format(toc-tic))

fig = plt.figure(figsize=(10, 8))
plt.imshow(det.assemble_image_stack(patternOp),vmin=0, vmax=5)
plt.title('Open state')
plt.show()
