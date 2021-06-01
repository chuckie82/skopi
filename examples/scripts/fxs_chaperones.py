import matplotlib.pyplot as plt
import skopi as sk

# Set up a square detector
# (no. of pixels per row, side length (m), sample-to-detector distance (m))
n_pixels, det_size, det_dist = (156, 0.1, 0.2)
det = sk.SimpleSquareDetector(n_pixels, det_size, det_dist)

# Set up x-ray beam
# photon energy: 4600eV
# photons per shot: 1e12 photons/pulse, 
# x-ray focus radius: 0.5e-6m
beam = sk.Beam("../input/beam/amo86615.beam")

# Set jet radius (m)
jet_radius = 1e-6

# Set up particle
# pdb file of lidless mmCpn in open state
particle = sk.Particle()
particle.read_pdb("../input/pdb/3iyf.pdb", ff='WK')

# Set up FXS experiment with 20 particles per shot
exp = sk.FXSExperiment(det, beam, jet_radius, [particle], n_part_per_shot=20)

# Generate an image
img = exp.generate_image()

# Visualize
plt.imshow(img, vmin=0, vmax=3, origin='lower'); 
plt.colorbar(); 
plt.show()
