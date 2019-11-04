import numpy as np
import sys
import Sinc_func
import new_profile_libsaxs


class RadialDistributionFunction:

    def __init__(self,bin_size=0.5,max_dist=10000):
        
        self.max_distance = max_dist
        self.bin_size = bin_size;
        self.one_over_bin_size = 1.0/ bin_size
        self.nbins = int(max_dist * self.one_over_bin_size) + 1
        self.values = np.zeros((self.nbins,1),dtype=np.float64)
         
    def reset(self):
        self.values = np.zeros((self.nbins,1),dtype=np.float64)
        
    def get_bin_size(self):
        return  self.bin_size
    
    def get_one_over_bin_size(self):
    
        return self.one_over_bin_size
        
    def get_values(self):
        return self.values
    
    def get_nbins(self):
    
        return self.nbins
    
    def get_max_distance(self):
    
        return self.max_distance

   
def add2distribution(rdist,distance,val):

    bin = int(rdist.get_one_over_bin_size() * distance)
    #print "Bin",bin,"\n"
    #print "rdistvalues_shape",rdist.values.shape,"\n"
    #print "Val",val,"\n"
    rdist.values[bin] += val
    return rdist

def radial_distributions_to_partials(p , ndists, r_dists):
    dd = []
    xx = []
    nbins = r_dists[0].get_nbins()
    delta_x = r_dists[0].get_bin_size()
    sf = Sinc_func.Sinc_func(np.sqrt(r_dists[0].get_max_distance()) * 3.0, 0.0001)
    for iq in range(p.nsamples):
        
        q = p.get_q(iq)
    
        for r in range(nbins):
            
            
            qd = r * delta_x * q  # r * delta_x = dist
            dd.append(qd)
            x = sf.sincc(qd/np.pi)
            xx.append(x)
            
            #xx.append(x)
            if r_dists[0].values[r] > 0.0:
      
                p.vac_vac[iq] += r_dists[0].values[r] * x;
                p.dum_dum[iq] += r_dists[1].values[r] * x;
                p.vac_dum[iq] += r_dists[2].values[r] * x;

            if ndists == 6:
        
                p.h2o_h2o[iq] +=  r_dists[3].values[r] * x
                p.vac_h2o[iq] +=  r_dists[4].values[r] * x
                p.dum_h2o[iq] +=  r_dists[5].values[r] * x
        
        
        modulation_function_parameter = 0.23
        scaling_factor = np.exp(-modulation_function_parameter * p.get_q(iq) * p.q[iq])
            
        p.vac_vac[iq] *= scaling_factor
        p.vac_dum[iq] *= scaling_factor
        p.dum_dum[iq] *= scaling_factor

        if ndists == 6:
    
            p.vac_h2o[iq] *= scaling_factor
            p.dum_h2o[iq] *= scaling_factor
            p.h2o_h2o[iq] *= scaling_factor
    
    return p
    
        
    

