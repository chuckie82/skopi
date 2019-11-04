import sys
import form_factor_table
import radial_distribution_function
import numpy as np

class Profile:

    def __init__(self,q_min,q_max,q_delta):
    
        self.nsamples = int(np.ceil((q_max - q_min) / q_delta))+1
        self.intensity = np.zeros((self.nsamples,1),dtype=np.float64)

        self.q = np.zeros((self.nsamples,1),dtype=np.float64)
        self.In = np.zeros((self.nsamples,1),dtype=np.float64)
        self.err = np.zeros((self.nsamples,1),dtype=np.float64)
        
        self.average_radius = 1.58
        self.average_volume_  = 17.5
        
        self.q[0] = q_min
        for i in range(1,self.nsamples):

            self.q[i] = self.q[i-1] + q_delta
        
        self.vacuum_ff = None

        self.dummy_ff = None
        self.rgyration = 0.0
        self.npartials = 0
        self.vac_vac = np.zeros((self.nsamples,1))
        self.vac_dum = np.zeros((self.nsamples,1),dtype=np.float64)
        self.dum_dum = np.zeros((self.nsamples,1),dtype=np.float64)
        self.vac_h2o = np.zeros((self.nsamples,1),dtype=np.float64)
        self.dum_h2o = np.zeros((self.nsamples,1),dtype=np.float64)
        self.h2o_h2o = np.zeros((self.nsamples,1),dtype=np.float64)

    def get_intensities(self):
        return self.In
    
    def get_q_size(self):
        
        return self.nsamples
    
    def get_max_q(self):
        
        return self.max_q
    
    def set_max_q(self,mq):
        
        self.max_q_ = mq

    def get_min_q(self):
        
        return self.min_q
    
    def set_min_q(self,mq):
        
        self.min_q = mq
    
    def get_delta_q(self):
        
        return self.delta_q
    
    def set_delta_q(self,dq):
        
        self.delta_q = dq
        
    def get_q(self,i):
    
        return self.q[i]
        
    def get_average_radius(self):
        
        return self.average_radius_
    
    def set_average_radius(self,ar):
        
        self.average_radius = ar
        
    def get_average_volume(self):

        return self.average_volume
    
    def set_average_volume(av):
    
        self.average_volume = av

    def saxs_profile_reset_In(self):

        self.In = np.zeros((self.nsamples,1),dtype=np.float64)

def calculate_profile_partial (prof,particles,saxs_sa,ft,vff,dff,ff_type='HEAVY_ATOMS'):
   
      
    r_size = 3
    wf = ft.get_water_form_factor()
    print wf
        
    coordinates = particles.get_atom_pos()
    print coordinates.shape

    elements = particles.get_atom_type()
    unique_elem = np.unique(elements)
    print unique_elem
    
    elements = elements.astype(np.int32)
   
    unique_elem = unique_elem.astype(np.int32)
 
    print("Start partial profile calculation for %d particles.\n " % len(coordinates))
    
    prof.vacuum_ff = np.zeros((len(coordinates),prof.nsamples),dtype=np.float64)

    prof.dummy_ff = np.zeros((len(coordinates),prof.nsamples),dtype=np.float64)

    water_ff = 0
    h2o_ff_i = 0
    h2o_ff_j = 0
        
    #vacuum_ff = form_factor_table_form_factors()
    #for m in range(len(coordinates)):
    #   vacuum_ff[m] = form_factor_table.get_vacuum_form_factors()
    #    dummy_ff[m] = form_factor_table.get_dummy_form_factors()
    for m in range(len(coordinates)):
        #for i in range(len(unique_elem)):
        #    if unique_elem[i] == elements[m]:
        #        # print m,unique_elem[i],'\n'
        #        break
        print "m=",m,"\n"
        if elements[m] == 6:
            i = 2
            
        #print vff[2,:]
        #sys.exit()
        if elements[m] == 7:
            i = 3
        if elements[m] == 8:
            i = 4
        if elements[m] == 15:
            i = 8
        if  elements[m] ==  16:
            i == 9
        if elements[m] == 17:
            i = 10
            
        #print unique_elem[i]
       
        #print elements[0].astype(np.int32)
        prof.vacuum_ff[m,:]   = vff[i,:]
        #print vacuum_ff[m,:]

        #print m,i,vff[i,0],'\n'
        prof.dummy_ff[m,:]   = dff[i,:]
            
           
        #print len(surface)
        #print len(coordinates)
                
    if len(saxs_sa) == len(coordinates):
        water_ff = np.resize(water_ff,(len(coordinates),1))
        r_size = 6
                    
        #print wf.shape
        for n in range(len(coordinates)):
            water_ff[n] = saxs_sa[n] * wf
                
    
    r_dist = []
    #mdist = scipy.spatial.distance.cdist(coordinates,coordinates)
    max_dist = calculate_max_distance(coordinates)
    
    #sf = Sinc_func.Sinc_func(np.sqrt(max_dist * 3.0), 0.0001)
    
    
    for i in range(r_size):
        r_dist.append(new_rdf_radial_distribution_function.RadialDistributionFunction(0.5,max_dist))
   
    for i in range(len(coordinates)):
        
        print i
        vac_ff_i = prof.vacuum_ff[i,0]
        dum_ff_i = prof.dummy_ff[i,0]
        if len(saxs_sa) == len(coordinates):
            h2o_ff_i = water_ff[i]
            
        for j in range(i+1,len(coordinates)):
            
                
            vac_ff_j = prof.vacuum_ff[j,0]
            dum_ff_j = prof.dummy_ff[j,0]
            
            dist = (coordinates[i,0]-coordinates[j,0])**2 + (coordinates[i,1]-coordinates[j,1])**2 + (coordinates[i,2]-coordinates[j,2])**2
            #print 2*vac_ff_i*vac_ff_j
                #sys.exit()
            r_dist[0] = new_rdf_radial_distribution_function.add2distribution(r_dist[0],dist,
                            2.0 * vac_ff_i * vac_ff_j) #  constant
            r_dist[1] = new_rdf_radial_distribution_function.add2distribution(r_dist[1],dist,
                                           2.0 * dum_ff_i * dum_ff_j) # c1^2
            r_dist[2] = new_rdf_radial_distribution_function.add2distribution(r_dist[2],dist,
                               2.0 * (vac_ff_i * dum_ff_j +
                                     vac_ff_j * dum_ff_i)) # -c1
            if len(saxs_sa) == len(coordinates):
                
                h2o_ff_j = water_ff * saxs_sa[j]
                    
                r_dist[3] = new_rdf_radial_distribution_function.add2distribution(r_dist[3], dist, 2.0 *   3.5 * 3.5) #h2o_ff_i * h2o_ff_j) # c2^2
                r_dist[4] = new_rdf_radial_distribution_function.add2distribution(r_dist[4],dist,
                                     2.0 * (vac_ff_i * 3.5 +
                                          vac_ff_j * 3.5)) # c2
                r_dist[5] =  new_rdf_radial_distribution_function.add2distribution(r_dist[5],dist,
                                     2.0 * (3.5 * dum_ff_j +
                                          3.5 * dum_ff_i))# -c1*c2
                
                # Autocorrelation
        r_dist[0] =  new_rdf_radial_distribution_function.add2distribution(r_dist[0],0,vac_ff_i * vac_ff_i)#  constant
        r_dist[1] = new_rdf_radial_distribution_function.add2distribution(r_dist[1],0,dum_ff_i * dum_ff_i) # c1^2
        r_dist[2] = new_rdf_radial_distribution_function.add2distribution(r_dist[2],0,2 * vac_ff_i * dum_ff_i)# -c1
        
        if len(saxs_sa) == len(coordinates):
            print "Hello\n"
            r_dist[3] = new_rdf_radial_distribution_function.add2distribution(r_dist[3],0,
                             h2o_ff_i * h2o_ff_i)
            r_dist[4] = new_rdf_radial_distribution_function.add2distribution(r_dist[4], 0,
                             2 * vac_ff_i * h2o_ff_i)
            r_dist[5] = new_rdf_radial_distribution_function.add2distribution(r_dist[5], 0,
                             2 * h2o_ff_i * dum_ff_i)
    new_prof = new_rdf_radial_distribution_function.radial_distributions_to_partials(prof,r_size,r_dist)
    intensity = sum_profile_partials(new_prof,1.0, 0.0)  #c1 = 1.0, c2 = 0.0
    
    return intensity


def sum_profile_partials(p, c1,c2):

    rm = p.average_radius
    coeff = -np.power(4.0 * np.pi/ 3.0, 3.0/2.0) * (c1 * c1 - 1.0) / (16*np.pi)
    coeff *= (rm * rm)
    npartials = p.npartials
    
    
    #Initialize profile
    p.saxs_profile_reset_In()
    
    # Add profiles
    
    for iq in range(p.nsamples):
    
        q = p.q[iq]
        G_q = (c1*c1*c1)*np.exp(coeff*p.q[iq]*p.q[iq])
        
        p.In[iq] += p.vac_vac[iq]+p.dum_dum[iq] * (G_q * G_q) + \
        p.vac_dum[iq] * (-G_q)
        
        if npartials == 6:
        
            p.In[iq] += \
            (p.h2o_h2o[iq] * (c2 * c2) + p.vac_h2o[iq] * (c2)   + p.dum_h2o[iq] * (-G_q * c2))
    I = p.In
    return I
                           

def calculate_max_distance(coordinates):

    distance = 0
    for i in range(len(coordinates)):
                       
        for j in range(len(coordinates)):
                       

            d = (coordinates[i,0]-coordinates[j,0])** 2 + (coordinates[i,1] - coordinates[j,1])**2 + (coordinates[i,2]-coordinates[j,2])**2
                       
            if d > distance:

                distance = d
                       
    return distance
                      
