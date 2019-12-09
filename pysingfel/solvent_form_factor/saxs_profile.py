import sys
import form_factor_table
import radial_distribution_function
import numpy as np
import scipy
import matplotlib.pyplot as plt


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
        
        #self.scaling_factor = np.exp(-0.23 * p.get_q(iq) * p.q[iq])
        self.average_radius = 1.58
        self.average_volume_  = 17.5
        self.experimental_ = False
        self.q[0] = q_min
        
        for i in range(1,self.nsamples):

            self.q[i] = self.q[i-1] + q_delta
        self.sf = np.exp(-0.23 * self.q * self.q)

        
        self.vacuum_ff = None

        self.dummy_ff = None
        
        self.rgyration = 0.0
        self.npartials = 0
        
        self.vac_vac = np.zeros((self.nsamples,1),dtype=np.float64)
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

    
    def add_errors(self):

        for i in range(self.nsamples):
            ra = np.abs(np.random.poisson(10) / 10.0 - 1.0) + 1.0

            # 3% of error, scaled by 5q + poisson distribution
            self.err[i] = 0.03 * self.In[i] * 5.0 * (self.q[i] + 0.001) * ra

            self.experimental_ = True
            self.In[i] += self.err[i]
            
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
              
               s3 = self.err[i]
               outFile.write("%10.8f" % s3)

           outFile.write("\n")

       outFile.close()


def assign_form_factors_2_profile(particles,prof,saxs_sa,vff,dff,ft,num_atoms,r_size,verbose):
    
    
       
    if verbose ==1:
        fp = open('data/atomtypes.txt','w')
        fv = open('data/vacuum_impPy.txt','w')
        fd = open('data/dummy_impPy.txt','w')
    
    prof.vacuum_ff = np.zeros((num_atoms,prof.nsamples),dtype=np.float64)

    prof.dummy_ff = np.zeros((num_atoms,prof.nsamples),dtype=np.float64)

    wf = ft.get_water_form_factor()
    water_ff = 0
    
    atomic_variant = particles.get_atomic_variant()
    symbols = particles.get_atomic_symbol()
    residue  = particles.get_residue()
    
    table = ft.get_ff_cm_dict()
    for m in range(num_atoms):
           if m % 1000 == 0:
              print m
           
           ret_type = ft.get_form_factor_atom_type(symbols[m],atomic_variant[m], residue[m])
           if verbose == 1:
               fp.write("%s %s %s %s\n" % (symbols[m],atomic_variant[m],residue[m],ret_type))
           #if m == 0 or m==9:
           #   ret_type = 'N'
           idx =   table[ret_type]
           
           prof.vacuum_ff[m,:]   = vff[idx,:]
           
           prof.dummy_ff[m,:]   = dff[idx,:]
           
           if verbose == 1:
               fv.write("%.6f\n" % prof.vacuum_ff[m,0])
               fd.write("%.6f\n" % prof.dummy_ff[m,0])

    if len(saxs_sa) == num_atoms:
        water_ff = np.resize(water_ff,(num_atoms,1))
        r_size = 6
                       
           
        for n in range(num_atoms):
            water_ff[n] = saxs_sa[n] * wf
               
    return prof, water_ff,r_size


def calculate_profile_partial (prof,particles,saxs_sa,ft,vff,dff,verbose,c1,c2,ff_type='HEAVY_ATOMS'):
    
    if verbose ==1:
        fp = open('data/atomtypes_ImpPy.txt','w')
        fv = open('data/vacuum_ImpPy.txt','w')
        fd = open('data/dummy_ImpPy.txt','w')
    
    r_size = 3
    
 
    #coordinates = particles.get_atom_pos()
    coordinates = particles.get_atom_struct()
    coordinates = np.transpose(coordinates)
    coordinates = coordinates[:,0:3]
    #print coordinates
    
    
    print("Start partial profile calculation for %d particles.\n " % len(coordinates))
    

    num_atoms = len(coordinates)
    
    prof, water_ff, r_size = assign_form_factors_2_profile(particles,prof,saxs_sa,vff,dff,ft,num_atoms,r_size,verbose)
   
    r_dist = []
    max_dist = calculate_max_distance(coordinates)
    for i in range(r_size):
       r_dist.append(radial_distribution_function.RadialDistributionFunction(0.5,max_dist))
       
    nbins = r_dist[0].get_nbins()
    
    cd = scipy.spatial.distance.cdist(coordinates,coordinates)
    
    max_dist = np.max(cd)
    
    r0 = np.zeros((nbins,1),dtype=np.float64)
    r1 = np.zeros((nbins,1),dtype=np.float64)
    r2 = np.zeros((nbins,1),dtype=np.float64)
    r3 = np.zeros((nbins,1),dtype=np.float64)
    r4 = np.zeros((nbins,1),dtype=np.float64)
    r5 = np.zeros((nbins,1),dtype=np.float64)

    bins = (r_dist[0].get_one_over_bin_size()*cd).astype(int)
    
    vacuum2 =  np.outer(prof.vacuum_ff[:,0],prof.vacuum_ff[:,0])
    dummy2  =  np.outer(prof.dummy_ff[:,0],prof.dummy_ff[:,0])
    vd = np.outer(prof.vacuum_ff[:,0],prof.dummy_ff[:,0])
    
    if r_size == 6:
       vach2o = np.outer(prof.vacuum_ff[:,0],water_ff)
       dumh2o = np.outer(prof.dummy_ff[:,0],water_ff)
       h2o2 = np.outer(water_ff,water_ff)
    
    r0[0] = np.sum(np.diag(vacuum2))
    r1[0] = np.sum(np.diag(dummy2))
    r2[0] = 2.0 * np.sum(np.diag(vd))
    
    if r_size == 6:
       r3[0] = np.sum(np.diag(h2o2))
       r4[0] = 2.0 * np.sum(np.diag(vach2o))
       r5[0] = 2.0 * np.sum(np.diag(dumh2o))

    #print r0[0]
    #print r1[0]
    #print r2[0]
    
    vacuum2 = np.triu(vacuum2,k=1)
    dummy2 = np.triu(dummy2,k=1)
    if r_size == 6:
       h2o2 = np.triu(h2o2,k=1)
    vd[np.diag_indices(vd.shape[0])] = 0
    
    if r_size == 6:
       vach2o[np.diag_indices(vach2o.shape[0])]
       dumh2o[np.diag_indices(dumh2o.shape[0])]

    if verbose == 1:
        fp.close()
        fv.close()
        fd.close()
        fdist = open('data/dist_impPy.txt','w')

        for a in range(len(coordinates)):
            for c in range(len(coordinates)):
        
                fdist.write("%.6f\n" % cd[a,c])
            
        fdist.close()

    flat = bins.ravel()
    vacuum2 = vacuum2.ravel()
    dummy2 = dummy2.ravel()
    vd = vd.ravel()

    if r_size == 6:
       h2o2 = h2o2.ravel()
       vach2o = vach2o.ravel()
       dumh2o = dumh2o.ravel()

    lin_idx = np.argsort(flat, kind='mergesort')
    
    sp = np.split(lin_idx, np.cumsum(np.bincount(flat)[:-1]))
    
    for b in range(0,nbins):
        print b
      
        if len(sp[b]) == 0:
           continue
        
        r0[b] += 2.0*np.sum(vacuum2[sp[b]])
        r1[b] += 2.0*np.sum(dummy2[sp[b]])
        r2[b] += 2.0*np.sum(vd[sp[b]])
        if r_size == 6:
           r3[b] += 2.0*np.sum(h2o2[sp[b]])
           r4[b] += 2.0*np.sum(vach2o[sp[b]])
	   r5[b] += 2.0*np.sum(dumh2o[sp[b]])
        #print r0[b]
        #print r1[b]
        #print r2[b]
    

    r_dist[0].values = r0
    r_dist[1].values = r1
    r_dist[2].values = r2
    
    if r_size == 6:
       r_dist[3].values = r3
       r_dist[4].values = r4
       r_dist[5].values = r5
    
    new_prof = radial_distribution_function.radial_distributions_to_partials(prof,r_size,r_dist,verbose)
    
    newpartials = np.hstack((new_prof.q, new_prof.vac_vac,new_prof.dum_dum, new_prof.vac_dum))
  
    if verbose == 1:
       np.savetxt('data/partials_ImpPy.txt',newpartials,fmt='%.6f',delimiter=' ' , newline='\n')
       
    intensity = sum_profile_partials(new_prof,c1, c2,verbose)  #c1 = 1.0, c2 = 0.0
    
    if verbose == 1:
       np.savetxt('data/intensity_ImpPy.txt',intensity,fmt='%.6f',delimiter=' ' , newline='\n')
       
    return intensity



def sum_profile_partials(p, c1,c2,verbose):
    verbose = 0
    rm = p.average_radius
    
    
    coeff = -np.power(4.0 * np.pi/ 3.0, 3.0/2.0) * (c1 * c1 - 1.0) / (16*np.pi)
    coeff *= (rm * rm)
    npartials = p.npartials
    
    
    #Initialize profile
    p.saxs_profile_reset_In()
    
    if verbose == 1:
        gqq = []
    
    # Add profiless
    for iq in range(p.nsamples):
    
        q = p.q[iq]
        G_q = (c1*c1*c1)

        if np.abs(coeff) > 1e-8:
           G_q *= np.exp(coeff*p.q[iq]*p.q[iq])

        if verbose == 1:
            gqq.append(G_q)

        p.In[iq] += p.vac_vac[iq]+p.dum_dum[iq] * (G_q * G_q) + \
        p.vac_dum[iq] * (-G_q)
        
        if npartials == 6:
        
            p.In[iq] += \
            (p.h2o_h2o[iq] * (c2 * c2) + p.vac_h2o[iq] * (c2)   + p.dum_h2o[iq] * (-G_q * c2))

    I = p.In
    
    if verbose  == 1:
	gqq = np.asarray(gqq)
        np.savetxt('data/gq_ImpPy.txt',  gqq,fmt='%.6f', delimiter =' ',newline='\n')

    return I
                           

def calculate_max_distance(coordinates):
    dist = scipy.spatial.distance.cdist(coordinates,coordinates)
  
    distance = np.max(dist)
    return distance
                      

