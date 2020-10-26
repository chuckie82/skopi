import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class ParticleCollection(object):
    """
    This class provides functionality for a generic particle simulation, with methods for
    adding particles, moving them in the simulation domain, checking for intersects and overlaps.
    It also calculates basic properties of the particle collection within the domain.
    """

    def __init__(self, max_pcles=1000, debug=False):
        """
        Initialize the simulation. The key parameter here is the maximum number of particles,
        in order to pre-allocate array space.
        """
        self.pos = np.zeros((max_pcles,3), dtype=np.float)
        self.idx = np.zeros(max_pcles, dtype=np.int)
        self.radius = np.zeros(max_pcles, dtype=np.float)
        self.volume = np.zeros(max_pcles, dtype=np.float)
        self.mass = np.zeros(max_pcles, dtype=np.float)
        self.density = np.zeros(max_pcles, dtype=np.float)
        self.count = 0
        self.agg_count = 0
        self.next_idx = 0
        self.debug = debug
        
    def info(self):
        """Prints particle collection information to the standard output."""
        
        print("number of particles: %d" % self.count)
        print("position array size: %d" % self.pos.shape[0])
        print("bounding box: %s" % str(self.get_bb()))

    def update(self):
        """"Updates internally calculated parameters, such as mass and volume."""
        
        self.volume = (4./3.)*np.pi*self.radius**3.
        self.mass = self.volume*self.density
        
    def scale(self, scale=1.):
        """
        Applies a scale multiplier to all positions and radii. 
        Mass, volume are also updated accordingly.
        """
    
        self.pos *= scale
        self.radius *= scale
        self.update()
        
    def __str__(self):
        """
        Returns a string with the number of particles and the bounding box size.
        """

        return "<ParticleCollection object containing %d particles>" % (self.count)

    def get_bb(self):
        """
        Return the bounding box of the simulation domain.
        """

        xmin = self.pos[:,0].min() - self.radius[np.argmin(self.pos[:,0])]
        xmax = self.pos[:,0].max() + self.radius[np.argmin(self.pos[:,0])]

        ymin = self.pos[:,1].min() - self.radius[np.argmin(self.pos[:,1])]
        ymax = self.pos[:,1].max() + self.radius[np.argmin(self.pos[:,1])]

        zmin = self.pos[:,2].min() - self.radius[np.argmin(self.pos[:,2])]
        zmax = self.pos[:,2].max() + self.radius[np.argmin(self.pos[:,2])]

        return (xmin, xmax), (ymin,ymax), (zmin, zmax)

    def bb_aspect(self):
        """
        Returns the aspect ratio X:Y:Z of the bounding box.
        """

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()
        xsize = xmax-xmin
        ysize = ymax-ymin
        zsize = zmax-zmin

        return (xsize, ysize, zsize)/min(xsize, ysize, zsize)
    
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

    def add(self, pos, radius, density=1., check=False):
        """
        Add a particle to the particle collection.
        If check=True the distance between the proposed particle and each other
        is checked so see if they overlap. If so, False is returned.
        """

        if check:
            if not self.check(pos, radius):
                return False

        if len(pos) != 3:
            print('ERROR: particle position should be given as an x,y,z tuple')
            return None

        radius = float(radius)

        self.pos[self.count] = np.array(pos)
        self.radius[self.count] = radius
        self.volume[self.count] = (4./3.)*np.pi*radius**3.
        self.density[self.count] = density
        self.mass[self.count] = self.volume[self.count]*density

        self.count += 1
        self.idx[self.count-1] = self.next_idx
        self.next_idx += 1

        return True

    def intersect(self, position, direction, closest=True):
        """
        Wrapper for line_sphere() that detects if the position passed is for a
        monomer or an aggregates and handles each case.
        """

        if type(position)==tuple:
            position = np.array(position)

        if len(position.shape)==2: # position is an array, i.e. an aggregate
            # loop over each monomer in the passed aggregate and check if it
            # intersects any of the monomers already in the domain

            max_dist = 10000. # TODO calculate a sensible value here
            pc_id = None
            hits = None
            monomer_pos = None

            for pos in position:

                ids, dist = self.line_sphere(pos, direction, closest=True, ret_dist=True)
                if dist is not None:
                    if dist < max_dist:
                        max_dist = dist
                        monomer_pos = pos
                        pc_id = ids # id of the particle collection agg

            if pc_id is not None:
                hit = monomer_pos + max_dist * direction # position of closest intersect
                return pc_id, max_dist, hit
            else:
                return None, None, None

        else:

            ids, hits = self.line_sphere(position, direction, closest=closest, ret_dist=False)

        return ids, hits

    def line_sphere(self, position, direction, closest=True, ret_dist=False):
        """
        Accepts a position and direction vector defining a line and determines which
        particles in the simulation intersect this line, and the locations of these
        intersections. If closest=True only the shortest (closest) intersect is
        returned, otherwise all values are given.
        If ret_dist=True then the distance from position to the hit will be returned,
        rather than the coordinates of the hit itself.
        """

        # calculate the discriminator using numpy arrays
        vector = position - self.pos[0:self.count]
        b = np.sum(vector * direction, 1)
        c = np.sum(np.square(vector), 1) - self.radius[0:self.count] * self.radius[0:self.count]
        disc = b * b - c

        # disc<0 when no intersect, so ignore those cases
        possible = np.where(disc >= 0.0)
        # complete the calculation for the remaining points
        disc = disc[possible]
        ids = self.idx[possible]

        if len(disc)==0:
            return None, None

        b = b[possible]
        # two solutions: -b/2 +/- sqrt(disc) - this solves for the distance along the line
        dist1 = -b - np.sqrt(disc)
        dist2 = -b + np.sqrt(disc)
        dist = np.minimum(dist1, dist2)

        # choose the minimum distance and calculate the absolute position of the hit
        hits = position + dist[:,np.newaxis] * direction

        if closest:
            if ret_dist:
                return ids[np.argmin(dist)], dist[np.argmin(dist)]
            else:
                return ids[np.argmin(dist)], hits[np.argmin(dist)]
        else:
            if ret_dist:
                return ids, dist
            else:
                return ids, hits


    def check(self, pos, radius):
        """
        Accepts a proposed particle position and radius and checks if this overlaps with any
        particle currently in the domain. Returns True if the position is acceptable or
        False if not.
        """

        if len(pos.shape)==2: # passed an aggregate

            if cdist(pos, self.pos[0:self.count]).min() < (radius.max() + self.radius[0:self.count].max()) > 0:
                # TODO does not properly deal with polydisperse systems
                if self.debug: print('Cannot add aggregate here!')
                return False
            else:
                return True

        else:

            if (cdist(np.array([pos]), self.pos[0:self.count+1])[0] < (radius + self.radius[0:self.count+1].max())).sum() > 0:
                if self.debug: print('Cannot add particle here!')
                return False
            else:
                return True

    def farthest(self):
        """
        Returns the centre of the particle farthest from the origin
        """

        return self.pos.max()
