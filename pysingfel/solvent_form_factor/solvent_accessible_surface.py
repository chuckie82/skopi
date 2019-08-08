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
        all_nn_indices = tree.query_radius(points,r=np.transpose(radius)) # NNs within distance of 1.5 of point
        #for j in range (100):
        #print "dist0",j,"=",self.distance(atoms,j,9),"\n"
        #for i in range(len(all_nn_indices)):
        #print all_nn_indices[i],'\n'


        #print all_nn_indices
        #print all_nn_indices[2],'\n'

        #ann = np.array(all_nn_indices,dtype=np.int32)
        #print ann
        #print(all_nn_indices[0])
        #print all_nn_indices.shape
        #all_nns = [[points[idx] for idx in nn_indices] for nn_indices in all_nn_indices]

        #n.append(all_nn_indices)

        #ann = np.array(an,dtype=np.int32)

        #sys.exit()
        return all_nn_indices

    '''

    for nns in all_nns:
        print(nns)
        neighbor_indices = []
        atom_k = atoms[k]
        radius =  atom_k[3] + probe + probe #atom_k.radius + probe + probe
        indices = range(k)
        indices.extend(range(k + 1, len(atoms)))
        for i in indices:
            atom_i = atoms[i]
            dist = np.sqrt((atom_k[0] - atom_i[0])**2 + (atom_k[1] - atom_i[1])**2 + (atom_k[2] - atom_i[2])**2)
            if dist < radius + atom_i[3]:
                neighbor_indices.append(i)
        return neighbor_indices
        
    '''

    def calculate_asa(self, atoms, probe=1.4, n_sphere_point=100):

        """
        Returns the accessible-surface areas of the atoms, by rolling a
        ball with probe radius over the atoms with their radius
        defined.
        """
        max_radius = 0.0

        test_point = np.zeros((3,1))
        sphere_points = self.generate_sphere_points(n_sphere_point)
        #sphere_points = self.gen_new_sphere_dots(probe,5.0)
        points = len(sphere_points)
        const = 4.0 * math.pi / len(sphere_points)
        areas = []
        fractions = []
        sas = []
        #print sphere_points
        sp = np.asarray(sphere_points)
        count = 0
        total_count = 0
        #atoms = atoms[0:20]
        mm = 0
        neighbor_indices = self.find_tree_neighbors(atoms,probe)
        #print neighbor_indices.shape


        #sys.exit()
        for i, atom_i in enumerate(atoms):
            if i % 1000 == 0:
                print(i)
            
            
            
            
            #neighbor_indices = self.find_neighbor_indices(atoms, probe, i)
            #neighbor_indices = self.find_tree_neighbors(atoms,probe,i)
            #print neighbor_indices.shape

            #sys.exit()
            #n_neighbor = len(neighbor_indices)
            #print "ni",i,"=",len(neighbor_indices[i]),'\n'

            n_neighbor = len(neighbor_indices[i])
            j_closest_neighbor = 0
            radius = atom_i[3] + max_radius + probe

            n_accessible_point = 0

            for point in sp:
                #print(len(sp))
                total_count += 1
                is_accessible = True
                point.reshape(3, 1)
                radius.reshape(1, 1)
                test_point.reshape(3,1)

                tp = point * radius

                test_point = np.array([tp[0] + atom_i[0], tp[1] + atom_i[1],tp[2] + atom_i[2]])
                #print "test_point ",i,"=",test_point,'\n'
                cycled_indices = range(j_closest_neighbor, n_neighbor)
                cycled_indices.extend(range(j_closest_neighbor))

                for j in cycled_indices:
                    #print "len_cycled_indices=",len(cycled_indices)

                    atom_j = atoms[neighbor_indices[i][j]]
                    #print atom_j.shape
                    #print "atom_j3=", atom_j[3],'\n'
                    r = atom_j[3] + probe
                    #print "r=",r,'\n'
                    #print "r2=",r*r,'\n'
                    xsq=(atom_j[0]-test_point[0])*(atom_j[0]-test_point[0])
                    ysq=(atom_j[1]-test_point[1])*(atom_j[1]-test_point[1])
                    zsq=(atom_j[2]-test_point[2])*(atom_j[2]-test_point[2])
                    
                    diffsq = xsq + ysq + zsq
                    #print "diff2=",diff*diff,'\n'
                    if diffsq < r * r:
                        j_closest_neighbor = j
                        is_accessible = False
                        break
                if is_accessible:
                    n_accessible_point += 1
            #print(n_accessible_point/len(sp))
            #print(n_accessible_point)
            area = const*n_accessible_point*radius*radius
            sa = 4*np.pi*radius * radius
            fraction = (float(n_accessible_point))/len(sp)
            areas.append(area)
            sas.append(sa)
            fractions.append(fraction)
            #print(n_accessible_point)
            count +=1
        f = np.array(fractions)
        a = np.array(areas)
        s = np.array(sas)
             
        return a,f,s

