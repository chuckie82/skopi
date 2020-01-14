import math
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KDTree,BallTree
import sys

class SolventAccessibleSurface(object):

    def __init__(self):

        print("Creating solvent accessible surface object")

    def find_neighbor_indices(self,atoms, probe,k):
        """
        Returns list of indices of atoms within probe distance to atom k.
        """

        neighbor_indices = []
        atom_k = atoms[k]
        radius =  atom_k[3] + probe + probe #atom_k.radius + probe + probe
        indices = range(k)
        indices.extend(range(k + 1, len(atoms)))
        for i in indices:
            atom_i = atoms[i]
            dist = math.sqrt((atom_k[0] - atom_i[0])**2 + (atom_k[1] - atom_i[1])**2 + (atom_k[2] - atom_i[2])**2)
            if dist < radius + atom_i[3]:
                neighbor_indices.append(i)
        print neighbor_indices
        return neighbor_indices

    def generate_sphere_points(self,n):

        """
        Returns list of coordinates on a sphere using the Golden-
        Section Spiral algorithm.
        """

        points = []
        inc = np.pi * (3 - np.sqrt(5))
        offset = 2 / float(n)

        for k in range(int(n)):
            y = k * offset - 1 + (offset / 2)
            r = np.sqrt(1 - y * y)
            phi = k * inc
            points.append([np.cos(phi) * r, y, np.sin(phi) * r])

        return points


    def gen_new_sphere_dots(self,atoms,probe=1.4,density=5.0):

        res = []

        num_equat = 2 * np.pi * radius * np.sqrt(density)
        vert_count = int(0.5 * num_equat)

        for i in range(vert_count):
            phi = (np.pi * i) / vert_count
            z = np.cos(phi)
            xy = np.sin(phi)
            horz_count = int(xy * num_equat)

            for j in range(horz_count - 1):
                theta = (2 * np.pi * j) / horz_count
                x = xy * np.cos(theta)
                y = xy * np.sin(theta)

        res.append([radius * x, radius * y, radius * z])

        return res


    def distance(self,atoms,i,j):
        atom_i = atoms[i]
        atom_j = atoms[j]

        return np.sqrt((atom_i[0]-atom_j[0])**2 + (atom_i[1]-atom_i[1])**2 + (atom_i[2]-atom_i[2])**2)

    def find_tree_neighbors(self,atoms,probe):
        

        points = []
        p = np.ones((len(atoms),1),dtype=np.int32)
        radius = atoms[:,3] + probe + probe

        for i in range(len(atoms)):
            points.append([atoms[i,0], atoms[i,1], atoms[i, 2]])
        tree = KDTree(points,leaf_size=2)
        print 'RADIUS=',radius,'\n'
        all_nn_indices = tree.query_radius(points,r=np.transpose(radius)) # NNs
        return all_nn_indices


    def calculate_asa(self, atoms, probe=1.4, n_sphere_point=100): # TODO: probe of water set as constant

        """
        Returns the accessible-surface areas of the atoms, by rolling a
        ball with probe radius over the atoms with their radius
        defined.
        """
        max_radius = 0.0
        # TODO: time each function and triple for loops
        test_point = np.zeros((3,1))
        sphere_points = self.generate_sphere_points(n_sphere_point) # TODO: return np.array 
        
        points = len(sphere_points)
        const = 4.0 * math.pi / len(sphere_points)
        areas = [] # TODO: initialize as np array
        fractions = []
        sas = []
        #print sphere_points
        sp = np.asarray(sphere_points)
        count = 0 # TODO: remove, not used
        total_count = 0
        
        
        neighbor_indices = self.find_tree_neighbors(atoms,probe) # TODO: return 2D np.array
        #print neighbor_indices.shape


        #sys.exit()
        for i, atom_i in enumerate(atoms):
            if i % 1000 == 0:
                print(i)
            
            n_neighbor = len(neighbor_indices[i])
            j_closest_neighbor = 0
            radius = atom_i[3] + max_radius + probe # TODO: this could move out of the for loop

            n_accessible_point = 0

            for point in sp: # for each sphere_points
                total_count += 1
                is_accessible = True
                point.reshape(3, 1) # TODO: let's check and remove reshape
                radius.reshape(1, 1)
                test_point.reshape(3,1) # TODO: get rid of this

                tp = point * radius # TODO: move out of for loop

                test_point = np.array([tp[0] + atom_i[0], tp[1] + atom_i[1],tp[2] + atom_i[2]])
                cycled_indices = range(j_closest_neighbor, n_neighbor) #TODO: Remove j_closest_neighbor and replace with 0
                print "cycled1: ", cycled_indices
                cycled_indices.extend(range(j_closest_neighbor)) # TODO: remove this line?
                print "cycled2: ", cycled_indices

                for j in cycled_indices:
                    atom_j = atoms[neighbor_indices[i][j]]
                    r = atom_j[3] + probe # TODO: take this out of triple for loop; r[j]
                    xsq=(atom_j[0]-test_point[0])*(atom_j[0]-test_point[0]) # TODO: express as one line np operation, xyzSq = np.sum((atom_j - test_point)**2,axis=0)
                    ysq=(atom_j[1]-test_point[1])*(atom_j[1]-test_point[1])
                    zsq=(atom_j[2]-test_point[2])*(atom_j[2]-test_point[2])                    
                    diffsq = xsq + ysq + zsq
                    if diffsq < r * r: # TODO: move outside for loop
                        j_closest_neighbor = j
                        is_accessible = False
                        break
                # TODO: element-wise compare operation for diffsq and rSq
                if is_accessible:
                    n_accessible_point += 1
            #print(n_accessible_point/len(sp))
            #print(n_accessible_point)
            area = const*n_accessible_point*radius*radius
            sa = 4*np.pi*radius * radius
            fraction = (float(n_accessible_point))/len(sp)
            areas.append(area) # TODO: change areas and sas and fractions to np array
            sas.append(sa)
            fractions.append(fraction) # This one is important
            count +=1 # TODO: remove, not used
        f = np.array(fractions) # TODO: no need if fractions is already np array
        a = np.array(areas)
        s = np.array(sas)
             
        return a,f,s

