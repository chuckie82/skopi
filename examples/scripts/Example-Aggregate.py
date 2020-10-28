import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

import pysingfel as ps
from pysingfel.particlePlacement import *

def drawSphere(xCenter, yCenter, zCenter, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

num = 5

input_dir='../input'
beamfile=input_dir+'/beam/amo86615.beam'
pdbfile=input_dir+'/pdb/3iyf.pdb'

beam = ps.Beam(beamfile)
particle = ps.Particle()
particle.read_pdb(pdbfile, ff='WK')

particles = {particle: num} 
part_states, part_positions = distribute_particles(particles, beam.get_focus()[0]/2, jet_radius=1e-4, gamma=1.)
radius = max_radius(particles)

x = []
y = []
z = []
for i in range(num):
    x.append(part_positions[i,0])
    y.append(part_positions[i,1])
    z.append(part_positions[i,2])
x = np.array(x)
y = np.array(y)
z = np.array(z)
r = np.ones(num)*radius

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
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
