import numpy as np
import sys
import saxs_profile

class RadialDistributionFunction:

    def __init__(self,bin_size=0.5,max_dist=50):
        
        self.max_distance = max_dist
        self.bin_size = bin_size;
        self.one_over_bin_size = 1.0/ bin_size
        
        
        self.nbins = int(max_dist * self.one_over_bin_size) + 1
        #a = np.arange(0,self.nbins)
        #aa = np.tile(a,(301,1))l
        
        #r = np.linspace(0.0,3.0,301)
        #r = np.transpose(r)
        #rr = np.tile(r,(self.nbins,1))
        #rr = rr.reshape((301,459))
        
        #rr = np.transpose(rr)
        #print rr[0:4,:]
        #self.dd = np.zeros((301,self.nbins),dtype=np.float64)
        
        #self.dd = aa *(self.bin_size/np.pi) * rr

        
        #self.sc = np.sinc(self.dd)
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
  
    nbins = r_dists[0].get_nbins()
    #print "nbins=",nbins
    
    delta_x = r_dists[0].get_bin_size()
    #print "delta_x=",delta_x

    verbose = 0
    #sincqd = []
    #qdsave = []
    #s = []
    if verbose == 1:
        fp = open("sinc_ImpPy.txt",'w')
        fqd = open("qd_ImpPy.txt",'w')
        fq = open("q_marc_ImpPy.txt",'w')
        fr0 = open("r0_ImpPy.txt",'w')
        fr1 = open("r1_ImpPy.txt",'w')
        fr2 = open("r2_ImpPy.txt",'w')
        fvv = open("vacvac_ImpPy.txt",'w')
        fdd = open("dumdum_ImpPy.txt",'w')
        fvd = open("vacdum_ImpPy.txt",'w')
    

    for iq in range(p.nsamples):
        
        q = p.get_q(iq)
        #fq.write("%.6f\n" % q)
        #print q
        #print "Nbins",nbins
        #sys.exit()
        for r in range(nbins):
            
            
            qd = r * delta_x * q # r * delta_x = dist
            #qdsave.append(qd)
            #x = r_dists[0].sc[iq,r]
            x = np.sinc(qd/np.pi)
            #sincqd.append(x)
            #print delta_x*q
            

            #x =  np.sinc(qd/np.pi)
              
           
            #print r_dists[0]
            #if r_dists[0].values[r] > 0.0:
                #r_dists[0].values[r]
            p.vac_vac[iq] += r_dists[0].values[r] * x
            p.dum_dum[iq] += r_dists[1].values[r] * x
            p.vac_dum[iq] += r_dists[2].values[r] * x
            if verbose == 1:
               fr0.write("%.6f\n" % r_dists[0].values[r])
               fr1.write("%.6f\n" % r_dists[1].values[r])
               fr2.write("%.6f\n" % r_dists[2].values[r])
            #if ndists == 6:
            #
            #    p.h2o_h2o[iq] +=  r_dists[3].values[r] * x
            #    p.vac_h2o[iq] +=  r_dists[4].values[r] * x
            #    p.dum_h2o[iq] +=  r_dists[5].values[r] * x
        

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
            
        #    p.vac_h2o[iq] *= scaling_factor
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
    #sincqd = np.asarray(sincqd)
    #qdsave = np.asarray(qdsave)
    #s = np.asarray(s)
    #np.savetxt('sinc_marc.txt',  sincqd,fmt='%.6f', delimiter =' ',newline='\n')
    #np.savetxt('qd_marc.txt',qdsave, fmt='%.6f', delimiter =' ',newline='\n')
    #np.savetxt('scaling_marc.txt',s,fmt='%.6f', delimiter =' ',newline='\n')
    if verbose == 1:
        fq.close()
        fr0.close()
        fr1.close()
        fr2.close()
        fvv.close()
        fdd.close()
        fvd.close()
    return p
    
        
    


