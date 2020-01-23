import unittest 
import numpy as np
import os
import sys
sys.path.append("../../..")
import pysingfel as ps

class LibSaxsCompareDataTest(unittest.TestCase):
    
    def setUp(self):
       
        print "Testing code against libsaxs...\n"
        ff_table_file = None
        min_q = 0.0
        max_q = 3.0
        delta_q = 0.01
        
        pdb = '../../../examples/input/SAXS_10atoms_mod.pdb'
        c1 = 1.0
        c2 = 0.0

        self.particles = ps.particle.Particle()
        self.particles.read_pdb(pdb,'CM')
        self.num_atoms = self.particles.get_num_atoms()
        self.r_size = 3
        self.saxs_sa = np.zeros((3,1),dtype=np.float64)
      
        self.ft = ps.solvent_form_factor.form_factor_table.FormFactorTable(ff_table_file,min_q,max_q,delta_q)

        self.vff = self.ft.get_vacuum_form_factors()
        self.dff = self.ft.get_dummy_form_factors()
        
        self.prof = ps.solvent_form_factor.saxs_profile.Profile(min_q,max_q,delta_q)
        self.q_entries = self.prof.get_all_q()
        self.intensity = np.zeros((self.q_entries.shape[0],1),dtype=np.float64)     

        self.prof, water_ff,r_size = ps.solvent_form_factor.saxs_profile.assign_form_factors_2_profile(self.particles,self.prof,self.saxs_sa,self.vff,self.dff,self.ft,self.num_atoms,self.r_size)

        self.intensity = ps.solvent_form_factor.saxs_profile.calculate_profile_partial(self.prof, self.particles, self.saxs_sa, self.ft, self.vff,self.dff, c1,c2)

    
    def test_vacuum_form_factor_data(self):
        print "Testing vacuum form factor data against libsaxs...\n"
        vls = np.loadtxt('../data/vacuum_ls.txt')
   
        vpy = self.prof.vacuum_ff[:,0]
        self.assertTrue(np.allclose(vls,vpy))


    def test_dummy_form_factor_data(self):

        print "Testing dummy form factor data against libsaxs...\n"
        dpy = self.prof.dummy_ff[:,0]
        dls = np.loadtxt('../data/dummy_ls.txt')
        
        self.assertTrue(np.allclose(dpy,dls))
    
    def test_partial_profile_data(self):
       
        partial = self.prof.get_partial_profiles().T
       
        print "Testing partial profile data against libsaxs...\n"
        vv_ls = np.loadtxt('../data/vac_vac_ls.txt')
        self.assertTrue(np.allclose(vv_ls,partial[0,:]))

        dd_ls = np.loadtxt('../data/dum_dum_ls.txt')
        self.assertTrue(np.allclose(dd_ls,partial[1,:]))

        vd_ls = np.loadtxt('../data/vac_dum_ls.txt')
        self.assertTrue(np.allclose(vd_ls,partial[2,:]))

    def test_sinc_data(self):
        print "Testing sinc data against libsaaxs...\n"
        sincls = np.loadtxt('../data/sinc_ls.txt')
        sincc = np.loadtxt('../data/sinc_impPy.txt')
    
        self.assertTrue(np.allclose(sincls,sincc))

    def test_distance_data(self):
        print "Testing atomic distance data against libsaxs...\n"
        dls = np.loadtxt('../data/dist_ls.txt')
        dist = np.loadtxt('../data/dist_impPy.txt')
        self.assertTrue(np.allclose(dls,dist))
        
    def test_qd_data(self):
        print "Testing qd data against libsaxs...\n"
        qdls = np.loadtxt('../data/qd_ls.txt')
        qd = np.loadtxt('../data/qd_impPy.txt')
        self.assertTrue(np.allclose(qdls,qd))
        
    def test_q_data(self):
        print "Testing q data against libsaxs...\n"
        qls = np.loadtxt('../data/q_ls.txt')
        self.q_entries = np.array(self.q_entries).T
        self.assertTrue(np.allclose(qls,self.q_entries))
    
    def test_Gq_data(self):
        print "Testing Gq data against libsaxs...\n"
        gqls = np.loadtxt('../data/gq_ls.txt')
        gq = np.loadtxt('../data/gq_impPy.txt')
        self.assertTrue(np.allclose(gqls,gq))
    
    def test_intensity_data(self):
        
        print "Testing intensity data against libsaxs...\n"
        ils = np.loadtxt('../data/intensity_ls.txt')
        self.intensity = np.array(self.intensity).T
        self.assertTrue(np.allclose(ils,self.intensity))
    
if __name__ == '__main__':
    unittest.main()
