import h5py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from skopi.util import symmpdb
from skopi.ff_waaskirf_database import *
import skopi.particle
from scipy.spatial import distance
from skopi.aggregate import build_bpca
from skopi.particleCollection import *


def max_radius(particles):
    radius_current = 0
    for particle in particles:
        radius_arr = particle.atom_pos - np.mean(particle.atom_pos, axis=0)
        for row in radius_arr:
            radius = np.sqrt(row[0]**2+row[1]**2+row[2]**2)
            if radius > radius_current:
                radius_current = radius
    radius_max = radius_current
    return radius_max


def distribute_particles(particles, beam_focus_radius, jet_radius, sticking=False): #beam_focus_radius = 10e-6 #jet_radius = 1e-4
    """
    Randomly distribute particles within the focus region.
    If sticking is turned on, particles are forced to aggregate into a single cluster.
    """
    state = []
    for particle in particles:
        for count in range(particles[particle]):
            state.append(particle)
    radius_max = max_radius(particles)
    N = sum(particles.values()) # total number of particles
    coords = np.zeros((N,3)) # initialize N*3 array
    if sticking is True:
        # generate a particle cluster
        agg = build_bpca(num_pcles=N, radius=radius_max)
        # set the center of the particle cluster to a 1*3 random position inside the volume illuminated by the beam (cylinder)
        agg_center = np.zeros((1,3))
        agg_center[0] = beam_focus_radius*np.sqrt(np.random.uniform(0,1))*np.cos(np.random.uniform(0,2*np.pi))
        agg_center[1] = beam_focus_radius*np.sqrt(np.random.uniform(0,1))*np.sin(np.random.uniform(0,2*np.pi))
        agg_center[2] = jet_radius*np.random.uniform(-1, 1)
        for i in range(N):
            coords[i,0] = agg_center[0]+agg.pos[i,0]
            coords[i,1] = agg_center[1]+agg.pos[i,1]
            coords[i,2] = agg_center[2]+agg.pos[i,2]
    else:
        # generate N*3 random positions inside the volume illuminated by the beam (cylinder)
        for i in range(N):
            coords[i,0] = beam_focus_radius*np.sqrt(np.random.uniform(0,1))*np.cos(np.random.uniform(0,2*np.pi))
            coords[i,1] = beam_focus_radius*np.sqrt(np.random.uniform(0,1))*np.sin(np.random.uniform(0,2*np.pi))
            coords[i,2] = jet_radius*np.random.uniform(-1, 1)
        # calculate N*N distance matrix
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        # collision detection check (<2 maxRadius)
        collision = dist_matrix < 2*radius_max
        checkList = [collision[i][j] for i in range(N) for j in range(N) if j > i]
        if any(item == True for item in checkList):
            distribute_particles(particles, beam_focus_radius, jet_radius)
    return state, coords


def position_in_3d(particles, beam_focus_radius, jet_radius):
    state, coords = distribute_particles(particles, beam_focus_radius, jet_radius)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for i in range(len(state)):
        x = np.concatenate([x,coords[i][0]])
        y = np.concatenate([y,coords[i][1]])
        z = np.concatenate([z,coords[i][2]])
    return x, y, z


def drawSphere(xCenter, yCenter, zCenter, r):
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
