import numpy as np
import skopi.particle
from scipy.spatial import distance
from skopi.aggregate import build_bpca
from skopi.particleCollection import *


def max_radius(particles):
    """
    Determine the radius of the widest particle in input set.

    :param particles: list of Particle objects
    :return radius_max: maximum radius of widest particle
    """
    radius_current = 0
    for particle in particles:
        radius_arr = particle.atom_pos - np.mean(particle.atom_pos, axis=0)
        for row in radius_arr:
            radius = np.sqrt(row[0]**2+row[1]**2+row[2]**2)
            if radius > radius_current:
                radius_current = radius
    radius_max = radius_current
    return radius_max


def random_positions_in_beam(n_positions, beam_radius, jet_radius):
    """
    Compute random positions in the cylindrical volume formed by the 
    intersection of the beam and (gas) jet. See:
    https://mathworld.wolfram.com/DiskPointPicking.html
    for description of selecting random points on a disk.

    :param n_positions: number of positions to return
    :param beam_radius: beam radius (radius of cylindrical volume)
    :param jet_radius: jet radius (half height of cylindrical volume)
    :return coords: coordinates array of shape [n_positions, 3] 
    """
    r = beam_radius * np.sqrt(np.random.uniform(0,1,size=n_positions))
    theta = np.random.uniform(0,2*np.pi,size=n_positions)
    x, y = r * np.cos(theta), r * np.sin(theta)
    z = jet_radius * np.random.uniform(-1,1,size=n_positions)
    coords = np.vstack((x,y,z)).T
    
    return coords


def distribute_particles(particles, beam_radius, jet_radius, sticking=False, iteration=0, max_iter=10): 
    """
    Randomly distribute particles within the focus region (volume intersection of the
    beam and jet). If sticking is turned on, particles are forced to aggregate into a 
    single cluster rather than dispersed.

    :param particles: dictionary of particle object:num_particles
    :param beam_radius: beam width
    :param jet_radius: radius of (gas) jet
    :param sticking: whether particles are aggregated, boolean
    :param iteration: number of times distribute_particles function has been called
    :param max_iter: maximum number of iterations allowed to prevent infinite recursion
    :return state: list of particles
    :return coords: coordinates of redistributed particles
    """
    # gather list of particle objects
    state = []
    for particle in particles:
        for count in range(particles[particle]):
            state.append(particle)
            
    # get particle coordinates for aggregate or disperse conditions
    radius_max = max_radius(particles)
    N = sum(particles.values()) # total number of particles

    if sticking is True:
        # generate a particle cluster and randomly recenter in cylindrical volume
        agg = build_bpca(num_pcles=N, radius=radius_max)
        agg_center = random_positions_in_beam(1, beam_radius, jet_radius)
        coords = agg.pos + agg_center
    
    else:
        # generate N*3 random positions inside the illumindated cylindrical volume
        coords = random_positions_in_beam(N, beam_radius, jet_radius)
        
        # if particles overlap, try again up to max_iter attempts
        dist_matrix = distance.cdist(coords, coords, 'euclidean')
        collision = dist_matrix < 2*radius_max
        checkList = [collision[i][j] for i in range(N) for j in range(N) if j > i]
        if any(item == True for item in checkList):
            if iteration == max_iter:
                print("Please check beam and jet radii; intersection volume appears quite small.")
                return
            else:
                distribute_particles(particles, beam_radius, jet_radius, sticking=False, iteration=iteration+1)
            
    return state, coords


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
    return x,y,z
