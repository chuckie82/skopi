import numpy as np
import sys
import saxs_profile

class RadialDistributionFunction:

    def __init__(self,bin_size=0.5,max_dist): # in Angstroms
        
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


def radial_distributions_to_partials(p , ndists, r_dists,modulation_function_parameter=0.23):

    """
    Calculates the partial profiles from the corresponding radial distribution functions
    :param p: Profile object
    :param ndist: number of radial distribution functions
    :param r_dist list of radial distribution functions
    :param modulation function parameter: I(q) = I(0) * exp(-b*q*q), where b is this constant
    :return p: Profile object with calculated partial profiles
    """
  
    nbins = r_dists[0].get_nbins()
    
    delta_x = r_dists[0].get_bin_size()


    for iq in range(p.nsamples):
        
        q = p.get_q(iq)

        for r in range(nbins):
            
            qd = r * delta_x * q # r * delta_x = dist
            x = np.sinc(qd/np.pi)
            
            p.vac_vac[iq] += r_dists[0].values[r] * x
            p.dum_dum[iq] += r_dists[1].values[r] * x
            p.vac_dum[iq] += r_dists[2].values[r] * x
            
            
            if ndists == 6:
            
               p.h2o_h2o[iq] +=  r_dists[3].values[r] * x
               p.vac_h2o[iq] +=  r_dists[4].values[r] * x
               p.dum_h2o[iq] +=  r_dists[5].values[r] * x
        

        #modulation_function_parameter = 0.23
        #scaling_factor = np.exp(-modulation_function_parameter * p.get_q(iq) * p.q[iq])
        #scaling_factor = p.sf[iq]
        #p.vac_vac[iq] *= scaling_factor
        #p.vac_dum[iq] *= scaling_factor
        #p.dum_dum[iq] *= scaling_factor
        #fvv.write("%.6f\n" % p.vac_vac[iq])
        #fdd.write("%.6f\n" % p.dum_dum[iq])
        #fvd.write("%.6f\n" % p.vac_dum[iq])
        #s.append(scaling_factor)
        #if ndists == 6:
        #    
        #   #p.vac_h2o[iq] *= scaling_factor
        #    p.dum_h2o[iq] *= scaling_factor
        #    p.h2o_h2o[iq] *= scaling_factor
        #fqd.write("%.6f\n" % qd)
        
        #fp.write("%.6f\n" % x)v
    scaling_factor = p.sf
    p.vac_vac *= scaling_factor
    p.vac_dum *= scaling_factor
    p.dum_dum *= scaling_factor
    
    if ndists == 6:
        
        p.vac_h2o *= scaling_factor
        p.dum_h2o *= scaling_factor
        p.h2o_h2o *= scaling_factor
 
    return p
    
