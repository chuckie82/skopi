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
            dist = np.sqrt((atom_k[0] - atom_i[0])**2 + (atom_k[1] - atom_i[1])**2 + (atom_k[2] - atom_i[2])**2)
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
            points.append([np.cos(phi) * r, r*y, np.sin(phi) * r])

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
        an = []

        points = []
        p = np.ones((len(atoms),1),dtype=np.int32)
        radius = atoms[:,3] + probe + probe

        for i in range(len(atoms)):
            points.append([atoms[i,0], atoms[i,1], atoms[i, 2]])
        tree = BallTree(points)
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
            radius = probe + atom_i[3] + max_radius

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
                    diff = np.sqrt((atom_j[0]-test_point[0])**2 + (atom_j[1] - test_point[1])**2 + (atom_j[2]-test_point[2])**2)
                    #print "diff2=",diff*diff,'\n'
                    if diff * diff < r * r:
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
            
            count +=1
        f = np.array(fractions)
        a = np.array(areas)
        s = np.array(sas)
             
        return a,f,s




    def make_boxes(self,a, d_max):

        """
        :param self:
        :param a:
        :param d_max:
        :return:

        Returns dictionary which keys are indecies of boxes (regions)
        with d_max length side and values
        are indicies of atoms belonging to these boxes
        """

        b = defaultdict(list)  # space divided into boxes
        for i in xrange(len(a)):
            atom = a[i]
            box_coor = tuple((int(math.floor(atom[0] / d_max)),int(math.floor(atom[1]/d_max)),int(math.floor(atom[2]/d_max))))
            b[box_coor].append(i)
        return b


    def add_bond(self,a, a1, a2, conn, d_max):
        """

        :param self:
        :param a:
        :param a1:
        :param a2:
        :param conn:
        :param d_max:
        :return:

        If distance between atoms a1 and a2 is less than d_max (neighboring atoms),
        add atoms a1 and a2 in adjacency list connected to each other
        """

        atom1 = a[a1]
        atom2 = a[a2]
        if ((atom1[0] - atom2[0])**2 + (atom1[1]-atom2[1])**2 + (atom1[2] - atom2[2])**2) <= d_max * d_max:  # connected
            conn[a1].append(a2)
            conn[a2].append(a1)


    def neighbor_atoms(self,b, box):

        """
        :param self:
        :param b:
        :param box:
        :return:

        Returns list of atoms from half of neighbouring boxes of the box
        another half is accounted when symmetric (opposite) boxes considered
        """

        na = []  # list for neighboring atoms
        x, y, z = box  # coordinates of the box
        # top layer consisting of 9 boxes
        if (x + 1, y + 1, z + 1) in b: na.extend(b[(x + 1, y + 1, z + 1)])
        if (x, y + 1, z + 1) in b: na.extend(b[(x, y + 1, z + 1)])
        if (x + 1, y, z + 1) in b: na.extend(b[(x + 1, y, z + 1)])
        if (x, y, z + 1) in b: na.extend(b[(x, y, z + 1)])
        if (x - 1, y + 1, z + 1) in b: na.extend(b[(x - 1, y + 1, z + 1)])
        if (x + 1, y - 1, z + 1) in b: na.extend(b[(x + 1, y - 1, z + 1)])
        if (x, y - 1, z + 1) in b: na.extend(b[(x, y - 1, z + 1)])
        if (x - 1, y, z + 1) in b: na.extend(b[(x - 1, y, z + 1)])
        if (x - 1, y - 1, z + 1) in b: na.extend(b[(x - 1, y - 1, z + 1)])
        # half of the middle layer excluding the box itself (4 boxes)
        if (x + 1, y + 1, z) in b: na.extend(b[(x + 1, y + 1, z)])
        if (x, y + 1, z) in b: na.extend(b[(x, y + 1, z)])
        if (x + 1, y, z) in b: na.extend(b[(x + 1, y, z)])
        if (x + 1, y - 1, z) in b: na.extend(b[(x + 1, y - 1, z)])
        return na


    def adjacency_list(self,a, d_max):

        """
        :param self:
        :param a:
        :param d_max:
        :return:

        Returns adjacency list from coordinate file
        in O(len(a)) time
        """

        b = self.make_boxes(a, d_max)  # put atoms into the boxes with dmax length side
        # now go on boxes and check connections inside 3x3 superboxes
        conn = [[] for i in xrange(len(a))]  # list of bond lengths each atom implicated
        for box in b:
            lb = len(b[box])
            for i in range(lb):
                a1 = b[box][i]
                # check possible connections inside the box
                for j in range(i + 1, lb):
                    a2 = b[box][j]
                    self.add_bond(a, a1, a2, conn, d_max)
                # check connections with atoms from neighbouring boxes
                na = self.neighbor_atoms(b, box)  # list of such atoms
                for a2 in na:
                    self.add_bond(a, a1, a2, conn, d_max)
        return conn


    def find_neighbor_indices_modified(self,atoms, indices, probe, k):
        """
        Returns list of indices of atoms within probe distance to atom k.
        """
        neighbor_indices = []
        atom_k = atoms[k]
        radius = atom_k[3] + probe + probe
        for i in indices:
            if i == k: continue
            atom_i = atoms[i]
            dist2 = (atom_k[0]-atom_i[0])**2 + (atom_k[1]-atom_i[1])**2 + (atom_k[2]-atom_i[2])**2 # ToAn
            if dist2 < (radius + atom_i[3]) ** 2:  # ToAn
                neighbor_indices.append(i)
        return neighbor_indices


    def calculate_asa_optimized(self,atoms, probe, n_sphere_point=960):
        """
        Returns the accessible-surface areas of the atoms, by rolling a
        ball with probe radius over the atoms with their radius
        defined.
        """
        max_radius = 3.0
        sphere_points = self.generate_sphere_points(n_sphere_point)

        const = 4.0 * math.pi / len(sphere_points)
        areas = []
        fractions = []
        sas = []
        
        neighbor_list = self.adjacency_list(atoms, 2 * (probe + max(atoms, key=lambda p: p[3])[3]))
        print "Before loop..."
        for i, atom_i in enumerate(atoms):
            print(i)
            neighbor_indices = [neig for neig in neighbor_list[i]]
            neighbor_indices = self.find_neighbor_indices_modified(atoms, neighbor_indices, probe,i)  # even further narrow diapazon
            n_neighbor = len(neighbor_indices)
            j_closest_neighbor = 0
            radius = probe + atom_i[3] + max_radius

            n_accessible_point = 0
            for point in sphere_points:
                is_accessible = True
                test_point = np.array([(radius*point[0]+atom_i[0]),(radius*point[1] + atom_i[1]), (radius*point[2] + atom_i[2])])

                cycled_indices = range(j_closest_neighbor, n_neighbor)
                cycled_indices.extend(range(j_closest_neighbor))

                for j in cycled_indices:
                    atom_j = atoms[neighbor_indices[j]]
                    r = atom_j[3] + probe
                    diff2 = (atom_j[0] - test_point[0])**2 + (atom_j[1] - test_point[1])**2 + (atom_j[2]-test_point[2])**2
                    if diff2 < r * r:
                        j_closest_neighbor = j
                        is_accessible = False
                        break
                if is_accessible:
                    n_accessible_point += 1

            area = const * n_accessible_point * radius * radius
            fraction = float(n_accessible_point)/len(sphere_points)
            sa = 4.0 * np.pi * radius * radius
            areas.append(area)
            fractions.append(fraction)
            sas.append(sa)

        return areas,fractions,sas
