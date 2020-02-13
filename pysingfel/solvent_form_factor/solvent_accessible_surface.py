import math
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KDTree,BallTree
import sys
import time 


class SolventAccessibleSurface(object):

    def __init__(self):

        print("Creating solvent accessible surface object")



    def generate_sphere_points(self,n):

        """
        Returns list of coordinates on a sphere using the Golden-
        Section Spiral algorithm.
        (http://xsisupport.com/2012/02/25/evenly-distributing-points-on-a-sphere-with-the-golden-sectionspiral/)
        :param n: the number of points
        :return points: the points n by 3 numpy array
        """
        points = np.zeros((n,3))
        inc = np.pi * (3 - np.sqrt(5)) #TODO: remove magic numbers
        offset = 2 / float(n)
        # points on a unit sphere at origin
        for k in range(int(n)):
            y = k * offset - 1 + (offset / 2)
            r = np.sqrt(1 - y * y)
            phi = k * inc
            points[k,:] = [np.cos(phi) * r, y, np.sin(phi) * r]
            
        return points


    def distance(self,atoms,i,j):
        atom_i = atoms[i]
        atom_j = atoms[j]

        return np.sqrt((atom_i[0]-atom_j[0])**2 + (atom_i[1]-atom_i[1])**2 + (atom_i[2]-atom_i[2])**2)


    def find_tree_neighbors(self,atoms,probe):
        """
        #TODO: need explaining what this does
        Finds the nearest neighbors of all atoms within some query radius
        Radius is different for each atom; therefore the query radius is not constant
        This function only needs to be called once.
          
        :param atoms: a numpy array of [natoms x 4] consisting of coordinates x,y,z and radius
        :param probe: the radius of the probe solvent molecule in Angstroms (water -> 1.4 A)
        :return all_nn_indices:  a list of neighbors for all atoms


        """


        tree = KDTree(atoms[:,0:3], leaf_size=2) # another option BallTree and default leaf size 40.
        radius = np.transpose(atoms[:,3] + 2*probe)
        all_nn_indices = tree.query_radius(atoms[:,0:3],r=radius) # NNs
        return all_nn_indices


    def calculate_asa(self, atoms, probe=1.4, n_sphere_point=100): # TODO: probe of water set as constant

        """
        Returns the accessible-surface areas of the atoms, by rolling a
        ball with probe radius over the atoms with their radius
        defined. Uses the Shrake-Rupley algorithm (1973)
        
        :param atoms: an array of all atoms with 4 columns each (x,y,z,radius)
        :param probe: the radius of the probe solvent molecule in Angstroms (water -> 1.4 A)
        :param n_sphere_points: the points to check accessibility on each atom's surface
        :return s: the surface areas of the atoms
        :return a: the accessible surface areas of the atoms
        :return f: the fraction of surface accessible for each atom   
 
        """
        atoms = np.asarray(atoms) # TODO: convert atoms to np.array as input
        max_radius = 0.0
        # TODO: time each function and triple for loops
        test_point = np.zeros((3,1))

        time_start_sphere_point = time.time()
        sphere_points = self.generate_sphere_points(n_sphere_point)
        time_end_sphere_point = time.time()

        points,_ = sphere_points.shape #len(sphere_points)
        const = 4.0 * math.pi / points
        areas = [] # TODO: initialize as np array
        fractions = []
        sas = []
        areas_ = np.zeros((len(atoms),)) 
        fractions_ = np.zeros_like(areas_)
        sas_ = np.zeros_like(areas_)
        
        time_start_tree_neighbor = time.time()
        neighbor_indices = self.find_tree_neighbors(atoms,probe)
        time_end_tree_neighbor = time.time()

        time_start_triple_loop = time.time()
        #radius_ = atoms[:,3] + (max_radius + probe)

        for i, atom_i in enumerate(atoms):
            if i % 1000 == 0:
                print(i)
            
            n_neighbor = len(neighbor_indices[i])
            j_closest_neighbor = 0
            radius = atom_i[3] + max_radius + probe # TODO: this could move out of the for loop

            n_accessible_point = 0
            time_start_double_loop = time.time()
            
            for point in sphere_points:
                is_accessible = True
                tp = point * radius # TODO: move out of for loop
                # take sphere point, scale by radius, and shift to atom's center
                test_point = np.array([tp[0] + atom_i[0], tp[1] + atom_i[1],tp[2] + atom_i[2]])
                
                cycled_indices = range(j_closest_neighbor, n_neighbor) #TODO: Remove j_closest_neighbor and replace with 0
                cycled_indices.extend(range(j_closest_neighbor)) # TODO: remove this line?
                
                #all_atom_j = atoms[neighbor_indices[i][cycled_indices]]
                #all_r = all_atom_j[:,3] + probe
                #all_diffsq = np.sum((all_atom_j[0:3] - np.tile(test_point,(len(cycled_indices))))**2)
                #print "### all diff: ", all_diffsq

                for j in cycled_indices:
                    atom_j = atoms[neighbor_indices[i][j]]
                    r = atom_j[3] + probe # TODO: take this out of triple for loop; r[j]
                    diffsq = np.sum((atom_j[0:3] - test_point)**2)

                    if diffsq < r * r: # TODO: move outside for loop
                        j_closest_neighbor = j
                        is_accessible = False
                        break
                # TODO: element-wise compare operation for diffsq and rSq
                if is_accessible:
                    n_accessible_point += 1

            area = const*n_accessible_point*radius*radius #  accessible surface area
            sa = 4*np.pi*radius * radius # surface area of atom
            fraction = (float(n_accessible_point))/len(sphere_points) # fraction of accessible surface area, the most used quantity
            areas.append(area) # TODO: change areas and sas and fractions to np array
            sas.append(sa)
            fractions.append(fraction) # This one is important
            #count +=1 # TODO: remove, not used
            time_end_double_loop = time.time()
        time_end_triple_loop = time.time()
        f = np.array(fractions) # TODO: no need if fractions is already np array
        a = np.array(areas)
        s = np.array(sas)
        print("Time to generate sphere points: %s\n" % (time_end_sphere_point - time_start_sphere_point))
        print("Time to generate KD neighbor tree: %s\n" % (time_end_tree_neighbor - time_start_tree_neighbor))
        print("Time for triple loop %s\n" % (time_end_triple_loop - time_start_triple_loop))
        print("Time for double loop %s\n" % (time_end_double_loop - time_start_double_loop))
        
        
              
        return a,f,s

