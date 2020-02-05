import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pysingfel as ps
from pysingfel.particlePlacement import max_radius, distribute_particles, position_in_3d

# Create a particle object
particleOp = ps.Particle()
particleOp.read_pdb('../input/3iyf.pdb', ff='WK')

particleCl = ps.Particle()
particleCl.read_pdb('../input/3j03.pdb', ff='WK')

# Load beam
beam = ps.Beam('../input/exp_chuck.beam')

x, y, z = position_in_3d(particles={particleOp:50,particleCl:10}, beam_focus_radius=beam.focus_xFWHM/2, jet_radius=1e-4)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x*1e9, y*1e9, z*1e9)
ax.set_xlabel('X [nm]')
ax.set_ylabel('Y [nm]')
ax.set_zlabel('Z [nm]')
ax.set_title('Chaperones Distribution in 3D Real Space')
ax.set_xlim3d(-beam.focus_xFWHM/2*1e9,beam.focus_xFWHM/2*1e9)
ax.set_ylim3d(-beam.focus_xFWHM/2*1e9,beam.focus_xFWHM/2*1e9)
ax.set_zlim3d(-1e5,1e5)
#ax.auto_scale_xyz([-beam.focus_xFWHM/2*1e9,beam.focus_xFWHM/2*1e9], [-beam.focus_xFWHM/2*1e9,beam.focus_xFWHM/2*1e9], [-1e5,1e5])
ax.set_aspect('equal')
plt.show()

