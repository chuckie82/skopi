import h5py
import numpy as np
import sys
from pysingfel.util import symmpdb
from pysingfel.ff_waaskirf_database import *
import pysingfel.particle
from scipy.spatial import distance

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


def distribute_particles(particles, beam_focus_radius, jet_radius): #beam_focus_radius = 10e-6 #jet_radius = 1e-4
    state = []
    for particle in particles:
        for count in range(particles[particle]):
            state.append(particle)
    radius_max = max_radius(particles)
    N = sum(particles.values()) # total number of particles
    coords = np.zeros((N,3)) # initialize N*3 array
    # generate N*3 random positions
    for i in range(N):
        coords[i,0] = beam_focus_radius*np.random.uniform(-1, 1)
        coords[i,1] = beam_focus_radius*np.random.uniform(-1, 1)
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
        x = np.concatenate([x,state[i].atom_pos[:,0]+coords[i][0]])
        y = np.concatenate([y,state[i].atom_pos[:,1]+coords[i][1]])
        z = np.concatenate([z,state[i].atom_pos[:,2]+coords[i][2]])
    return x, y, z
