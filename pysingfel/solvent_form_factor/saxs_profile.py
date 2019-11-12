import sys
import form_factor_table
import radial_distribution_function
import numpy as np

class Profile:

    def __init__(self,q_min,q_max,q_delta):
        self.q_min = q_min
        self.q_max = q_max
        self.q_delta = q_delta
        
        
        self.nsamples = int(np.ceil((q_max - q_min) / q_delta))+1
        self.intensity = np.zeros((self.nsamples,1),dtype=np.float64)

        self.q = np.zeros((self.nsamples,1),dtype=np.float64)
        self.In = np.zeros((self.nsamples,1),dtype=np.float64)
        self.err = np.zeros((self.nsamples,1),dtype=np.float64)
        
        self.average_radius = 1.58
        self.average_volume_  = 17.5
        self.experimental_ = False
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

    def write_partial_profiles(self,file_name):
  
        
        print 'hello write_partial_profiles'
        try:
            fp = open(file_name,'w')
        except IOError as ioe:
            print("Can't open file %s: %s" % (file_name, ioe))
        else:
            print("Opened file for writing successfully.")


        #header line
        fp.write("# SAXS profile: number of points = %d,  q_min = %f, q_max = %f, delta_q=%f\n" % (self.nsamples,self.q_min,self.q_max,self.q_delta))

        
        fp.write("#    q    intensity \n")
            
        for i in range(self.nsamples):
                
                
            w1 = self.q[i]
            fp.write("%10.5f " % w1)
               
            fp.write("%15.8f " % self.vac_vac[i])
            fp.write("%15.8f " % self.dum_dum[i])
            fp.write("%15.8f " % self.vac_dum[i])
            fp.write("%15.8f " % self.dum_dum[i])
            fp.write("%15.8f " % self.vac_h2o[i])
            fp.write("%15.8f " % self.dum_h2o[i])
            fp.write("%15.8f " %self.h2o_h2o[i])
            fp.write("\n")

        fp.write("\n")
        fp.close()


    def write_SAXS_file(self,file_name,max_q=3.0):

       try:
           outFile = open(file_name,'w')
       
       except IOError as ioe:
           print("Can't open file %s: %s\n" % (file_name, ioe))

       # header line
       outFile.write("# SAXS profile: number of points = %d, q_min = %f, q_max = %f " % (self.nsamples,self.q_min,self.q_max))

       #if max_q > 0:
       #    outFile.write(str(max_q))
       #else:
       outFile.write(str(self.q_max))
       outFile.write(", delta_q =%f\n" % self.q_delta)
       outFile.write("#    q    intensity ")

       if self.experimental_:
            outFile.write("   error")
       outFile.write("\n")

       # Main data
       for i in range(self.nsamples):
           #print self.nsamples
           if self.q_max > 0 and self.q[i] > self.q_max:
               break
       
           s1 = self.q[i]
           #print s1

           outFile.write("%10.8f " % s1)

           s2 = self.In[i]
           #print s2
           outFile.write("%15.8f " % s2)

           if self.experimental_: # do not print error for theoretical profiles
              
               s3 = self.error_[i]
               outFile.write("%10.8f" % s3)

           outFile.write("\n")

       outFile.close()

def calculate_profile_partial (prof,particles,saxs_sa,ft,vff,dff,ff_type='HEAVY_ATOMS'):

    r_size = 3
    wf = ft.get_water_form_factor()
    print wf
    
    #coordinates = particles.get_atom_pos()
    coordinates = particles.get_atom_struct()
    coordinates = np.transpose(coordinates)
    coordinates = coordinates[:,0:3]
    #print coordinates
    
    
    
   

    #elements = particles.get_atom_type()
    #unique_elem = np.unique(elements)
    #print unique_elem
    
    e = particles.get_element()
    
    #elements = elements.astype(np.int32)
    
    
    symbols = particles.get_atomic_symbol()
    #atomic_variant = particles.get_atomic_variant()
    residue  = particles.get_residue()
    #ret_type = ft.get_form_factor_atom_type(symbols,e, residue)
    table = ft.get_ff_cm_map()
    #idx =   table[ret_type]
    #unique_elem = unique_elem.astype(np.int32)
 
    print("Start partial profile calculation for %d particles.\n " % len(coordinates))
    
    prof.vacuum_ff = np.zeros((len(coordinates),prof.nsamples),dtype=np.float64)

    prof.dummy_ff = np.zeros((len(coordinates),prof.nsamples),dtype=np.float64)

    water_ff = 0
    h2o_ff_i = 0
    h2o_ff_j = 0
    
    for i in range(24):
        vff[i,-1] = 0.0
        dff[i,-1] = 0.0
    #vacuum_ff = form_factor_table_form_factors()
    #for m in range(len(coordinates)):
    #   vacuum_ff[m] = form_factor_table.get_vacuum_form_factors()
    #    dummy_ff[m] = form_factor_table.get_dummy_form_factors()
#
    for m in range(len(coordinates)):
        print m
        
        ret_type = ft.get_form_factor_atom_type(symbols[m],e[m], residue[m])
        idx =   table[ret_type]
        #print ret_type
        #print idx
        #sys.exit()
        #for i in range(len(unique_elem)):
        #    if unique_elem[i] == elements[m]:
        #        # print m,unique_elem[i],'\n'
        #        break
        #print "m=",m,"\n"

        #print unique_elem[i]
       
        #print elements[0].astype(np.int32)
        prof.vacuum_ff[m,:]   = vff[idx,:] #vff[i,:]
        #print prof.vacuum_ff[m,:]

        #print m,i,vff[i,0],'\n'
        prof.dummy_ff[m,:]   = dff[idx,:]#dff[i,:]
        #print prof.dummy_ff[m,:]
        #sys.exit()
        #print idx
        
            
        
        #print len(surface)
    #print coordinates
    #sys.exit()
    if len(saxs_sa) == len(coordinates):
        water_ff = np.resize(water_ff,(len(coordinates),1))
        r_size = 6
                    
        #print wf.shape
        for n in range(len(coordinates)):
            water_ff[n] = saxs_sa[n] * wf
                
    r_dist = []
    #mdist = scipy.spatial.distance.cdist(coordinates,coordinates)
    max_dist = calculate_max_distance(coordinates)
    print max_dist
    #sys.exit()
    
    
    for i in range(r_size):
        r_dist.append(radial_distribution_function.RadialDistributionFunction(0.5,max_dist))
   
    for i in range(len(coordinates)):
        
        print i
        vac_ff_i = prof.vacuum_ff[i,0]
        dum_ff_i = prof.dummy_ff[i,0]
        if len(saxs_sa) == len(coordinates):
        
            h2o_ff_i = wf * saxs_sa[i]
            
        for j in range(i+1,len(coordinates)):
            
            vac_ff_j = prof.vacuum_ff[j,0]
            dum_ff_j = prof.dummy_ff[j,0]
            
            dist = (coordinates[i,0]-coordinates[j,0])**2 + (coordinates[i,1]-coordinates[j,1])**2 + (coordinates[i,2]-coordinates[j,2])**2
            
            r_dist[0] = radial_distribution_function.add2distribution(r_dist[0],dist,
                            2.0 * vac_ff_i * vac_ff_j) #  constant
                            
            r_dist[1] = radial_distribution_function.add2distribution(r_dist[1],dist,
                                           2.0 * dum_ff_i * dum_ff_j) # c1^2
                                           
            r_dist[2] = radial_distribution_function.add2distribution(r_dist[2],dist,
                               2.0 * (vac_ff_i * dum_ff_j +
                                     vac_ff_j * dum_ff_i)) # -c1
            if len(saxs_sa) == len(coordinates):
                
                h2o_ff_j = wf * saxs_sa[j]
                    
                r_dist[3] = radial_distribution_function.add2distribution(r_dist[3], dist, 2.0 *   h2o_ff_i * h2o_ff_j)  # c2^2
                
                r_dist[4] = radial_distribution_function.add2distribution(r_dist[4],dist,
                                     2.0 * (vac_ff_i * h2o_ff_j +
                                          vac_ff_j * h2o_ff_i)) # c2
                r_dist[5] =  radial_distribution_function.add2distribution(r_dist[5],dist,
                                     2.0 * (h2o_ff_i * dum_ff_j +
                                          h2o_ff_j * dum_ff_i))# -c1*c2
                
                # Autocorrelation
        r_dist[0] =  radial_distribution_function.add2distribution(r_dist[0],0,vac_ff_i * vac_ff_i)#  constant
        r_dist[1] = radial_distribution_function.add2distribution(r_dist[1],0,dum_ff_i * dum_ff_i) # c1^2
        r_dist[2] = radial_distribution_function.add2distribution(r_dist[2],0,2 * vac_ff_i * dum_ff_i)# -c1
        
        if len(saxs_sa) == len(coordinates):
            
            r_dist[3] = radial_distribution_function.add2distribution(r_dist[3],0,
                             h2o_ff_i * h2o_ff_i)
            r_dist[4] = radial_distribution_function.add2distribution(r_dist[4], 0,
                             2 * vac_ff_i * h2o_ff_i)
            r_dist[5] = radial_distribution_function.add2distribution(r_dist[5], 0,
                             2 * h2o_ff_i * dum_ff_i)
    new_prof = radial_distribution_function.radial_distributions_to_partials(prof,r_size,r_dist)
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
                      
