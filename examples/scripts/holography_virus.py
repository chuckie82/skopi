# Holography simulation:
# Generates a diffraction pattern of a virus with a gold reference particle
import matplotlib.pyplot as plt
import numpy as np
import skopi as sk

# Set up a square detector
# (no. of pixels per row, side length (m), sample-to-detector distance (m))
n_pixels, det_size, det_dist = (156, 0.1, 0.2)
det = sk.SimpleSquareDetector(n_pixels, det_size, det_dist)

# Set up x-ray beam
# photon energy: 4600eV
# photons per shot: 1e12 photons/pulse
# x-ray focus radius: 0.5e-6m
beam = sk.Beam("../input/beam/amo86615.beam")

# Set jet radius (m) of the particle injector
# Currently scattering from the jet is not taken into account in the diffraction simulation
jet_radius = 1e-6

# Set up particle
# pdb file of panleukopenia virus
particle = sk.Particle()
particle.read_pdb("../input/pdb/1fpv.pdb", ff='WK')

# Set up gold reference particle
xyz = np.loadtxt("../input/pdb/AuBall.xyz")
rparticle = sk.Particle()
rparticle.create_from_atoms([("AU", xyz[i]) for i in range(xyz.shape[0])])

# Set up holography experiment
exp = sk.HOLOExperiment(det, beam, [rparticle], [particle], jet_radius=jet_radius, ref_jet_radius=jet_radius)

# Generate an image
img = exp.generate_image()

# Visualize
plt.imshow(img, vmin=0, vmax=3, origin='lower');
plt.title("Holography: photons diffracted from panleukopenia virus with gold reference")
plt.colorbar() 
plt.show()
