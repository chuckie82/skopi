import sys
import form_factor_table
import radial_distribution_function
import numpy as np


class Profile:

    def __init__(self, q_min=0, q_max=3.0, q_delta=0.01):

        self.nsamples = int(np.ceil((q_max - q_min) / q_delta)) + 1
        self.intensity = np.zeros((self.nsamples, 1), dtype=np.float64)

        self.q = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.In = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.err = np.zeros((self.nsamples, 1), dtype=np.float64)

        self.average_radius = 1.58
        self.average_volume_ = 17.5
        self.partial_profiles = None

        self.q[0] = q_min
        for i in range(1, self.nsamples):
            self.q[i] = self.q[i - 1] + q_delta

        self.vacuum_ff = None

        self.dummy_ff = None
        self.rgyration = 0.0
        self.npartials = 0
        self.vac_vac = np.zeros((self.nsamples, 1))
        self.vac_dum = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.dum_dum = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.vac_h2o = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.dum_h2o = np.zeros((self.nsamples, 1), dtype=np.float64)
        self.h2o_h2o = np.zeros((self.nsamples, 1), dtype=np.float64)


    def get_vac_vac(self):

        return self.vac_vac

    def get_vac_dum(self):

        return self.vac_dum

    def get_dum_dum(self):

        return self.dum_dum

    def get_vac_h2o(self):

        return self.vac_h2o

    def get_dum_h2o(self):

        return self.dum_h2o

    def get_h2o_h2o(self):

        return self.h2o_h2o




    def get_intensities(self):
        return self.In

    def get_q_size(self):
        return self.nsamples

    def get_max_q(self):
        return self.max_q

    def set_max_q(self, mq):
        self.max_q_ = mq

    def get_min_q(self):
        return self.min_q

    def set_min_q(self, mq):
        self.min_q = mq

    def get_delta_q(self):
        return self.delta_q

    def set_delta_q(self, dq):
        self.delta_q = dq

    def get_q(self, i):
        return self.q[i]

    def get_average_radius(self):
        return self.average_radius_

    def set_average_radius(self, ar):
        self.average_radius = ar

    def get_average_volume(self):
        return self.average_volume

    def set_average_volume(av):
        self.average_volume = av

    def saxs_profile_reset_In(self):
        self.In = np.zeros((self.nsamples, 1), dtype=np.float64)

    def get_vacuum_ff(self):

        return self.vacuum_ff

    def get_dummy_ff(self):

        return self.dummy_ff

    def get_partial_profiles(self):

        return self.partial_profiles


def assignFormFactors(particles,prof,vff,dff,coordinates):
    
    #coordinates = particles.get_atom_pos()
    coordinates = particles.get_atom_struct()
    coordinates = np.transpose(coordinates)
    coordinates = coordinates[:,0:3]
    #print coordinates
     
    av = particles.get_atomic_variant()
     
    symbols = particles.get_atomic_symbol()
    residue  = particles.get_residue()
    table = ft.get_ff_cm_dict()

    # for testing purposes only, IMP last q has form factor zero
    # does not change result
    for i in range(vff.shape[0]):
       vff[i,-1] = 0.0
       dff[i,-1] = 0.0
    
    print("Start partial profile calculation for %d particles.\n" % len(coordinates))

    prof.vacuum_ff = np.zeros((len(coordinates), prof.nsamples), dtype=np.float64)

    prof.dummy_ff = np.zeros((len(coordinates), prof.nsamples), dtype=np.float64)

    for m in range(len(coordinates)):
       print m
            
       ret_type = ft.get_form_factor_atom_type(symbols[m],av[m], residue[m])
       idx =   table[ret_type]

       prof.vacuum_ff[m,:]   = vff[idx,:] #vff[i,:]
       #print prof.vacuum_ff[m,:]

       #print m,i,vff[i,0],'\n'
       prof.dummy_ff[m,:]   = dff[idx,:]
           
    return prof


def init_water_form_factor(surf_area,coordinates,ft):

    r_size = 3
    water_ff = 0
    wf = ft.get_water_form_factor()
    water_ff = 0

    if len(surf_area) == len(coordinates):
        water_ff = np.resize(water_ff, (len(coordinates), 1))
        r_size = 6

        # print wf.shape
        for n in range(len(coordinates)):
            water_ff[n] = surf_area[n] * wf

    return water_ff, r_size

def build_radial_distribution(prof,ft, surf_area,coordinates,water_ff,r_size):

    #print prof.get_vacuum_ff()

    wf = ft.get_water_form_factor()

    h2o_ff_i = 0
    h2o_ff_j = 0

    max_dist = calculate_max_distance(coordinates)

    r_dist = []
    for i in range(r_size):
        r_dist.append(radial_distribution_function.RadialDistributionFunction(0.5, max_dist))


    for i in range(len(coordinates)):

        #print(prof.get_vacuum_ff())

        vac_ff_i = prof.vacuum_ff[i, 0]
        dum_ff_i = prof.dummy_ff[i, 0]

        if len(surf_area) == len(coordinates):
            h2o_ff_i = wf * surf_area[i]

        for j in range(i + 1, len(coordinates)):

            vac_ff_j = prof.vacuum_ff[j, 0]
            dum_ff_j = prof.dummy_ff[j, 0]

            dist = (coordinates[i, 0] - coordinates[j, 0]) ** 2 + (coordinates[i, 1] - coordinates[j, 1]) ** 2 + (
                        coordinates[i, 2] - coordinates[j, 2]) ** 2

            r_dist[0] = radial_distribution_function.add2distribution(r_dist[0], dist,
                                                         2.0 * vac_ff_i * vac_ff_j)  # constant


            r_dist[1] = radial_distribution_function.add2distribution(r_dist[1], dist,
                                                         2.0 * dum_ff_i * dum_ff_j)  # c1^2
            r_dist[2] = radial_distribution_function.add2distribution(r_dist[2], dist,
                                                         2.0 * (vac_ff_i * dum_ff_j +
                                                                vac_ff_j * dum_ff_i))  # -c1
            if len(surf_area) == len(coordinates):
                h2o_ff_j = wf * surf_area[j]

                r_dist[3] = radial_distribution_function.add2distribution(r_dist[3], dist,  2.0 * h2o_ff_i * h2o_ff_j) #  # c2^2
                r_dist[4] = radial_distribution_function.add2distribution(r_dist[4], dist, 2.0 * (vac_ff_i * h2o_ff_j + vac_ff_j * h2o_ff_i)) # c2
                r_dist[5] = radial_distribution_function.add2distribution(r_dist[5], dist, 2.0 * (h2o_ff_i * dum_ff_j +  h2o_ff_j * dum_ff_i))  # -c1*c2

        # Autocorrelation
        r_dist[0] = radial_distribution_function.add2distribution(r_dist[0], 0, vac_ff_i * vac_ff_i)  # constant
        r_dist[1] = radial_distribution_function.add2distribution(r_dist[1], 0, dum_ff_i * dum_ff_i)  # c1^2
        r_dist[2] = radial_distribution_function.add2distribution(r_dist[2], 0, 2.0 * vac_ff_i * dum_ff_i)  # -c1

        if len(surf_area) == len(coordinates):

            r_dist[3] = radial_distribution_function.add2distribution(r_dist[3], 0, h2o_ff_i * h2o_ff_i)
            r_dist[4] = radial_distribution_function.add2distribution(r_dist[4], 0, 2.0 * vac_ff_i * h2o_ff_i)
            r_dist[5] = radial_distribution_function.add2distribution(r_dist[5], 0, 2.0 * h2o_ff_i * dum_ff_i)

    return r_dist


def calculate_max_distance(coordinates):

    distance = 0
    for i in range(len(coordinates)):

        for j in range(len(coordinates)):

            d = (coordinates[i, 0] - coordinates[j, 0]) ** 2 + (coordinates[i, 1] - coordinates[j, 1]) ** 2 + (
                        coordinates[i, 2] - coordinates[j, 2]) ** 2

            if d > distance:
                distance = d

    return distance


def sum_profile_partials(p, c1, c2):
    rm = p.average_radius
    coeff = -np.power(4.0 * np.pi / 3.0, 3.0 / 2.0) * (c1 * c1 - 1.0) / (16 * np.pi)
    coeff *= (rm * rm)
    npartials = p.npartials

    # Initialize profile
    p.saxs_profile_reset_In()

    # Add profiles

    for iq in range(p.nsamples):


        q = p.q[iq]
        G_q = (c1 * c1 * c1) * np.exp(coeff * p.q[iq] * p.q[iq])

        p.In[iq] += p.vac_vac[iq] + p.dum_dum[iq] * (G_q * G_q) + \
                    p.vac_dum[iq] * (-G_q)

        if npartials == 6:
            p.In[iq] += \
                (p.h2o_h2o[iq] * (c2 * c2) + p.vac_h2o[iq] * c2 + p.dum_h2o[iq] * (-G_q * c2))
    I = p.In
    return I

