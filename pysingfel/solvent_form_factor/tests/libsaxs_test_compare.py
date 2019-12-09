import unittest 
import numpy as np
import os
import sys

class LibSaxsCompareDataTest(unittest.TestCase):
    
    def setUp(self):
        print "Testing code against libsaxs...\n"
        
    def test_partial_profile_data(self):

        print "Testing partial profile data against libsaxs...\n"
        vv_ls = np.loadtxt('../data/vac_vac_ls.txt')
        vv_impPy = np.loadtxt('../data/vacvac_impPy.txt')
        self.assertTrue(np.allclose(vv_ls,vv_impPy))

        dd_ls = np.loadtxt('../data/dum_dum_ls.txt')
        dd_impPy = np.loadtxt('../data/dumdum_impPy.txt')
        self.assertTrue(np.allclose(dd_ls,dd_impPy))

        vd_ls = np.loadtxt('../data/vac_dum_ls.txt')
        vd_impPy = np.loadtxt('../data/vacdum_impPy.txt')
        self.assertTrue(np.allclose(vd_ls,vd_impPy))
    """
    def test_radial_distribution_data(self):

        rd0 = np.loadtxt('data/rdist0_ls.txt')
        rd1 = np.loadtxt('data/rdist1_ls.txt')
        rd2 = np.loadtxt('data/rdist2_ls.txt')

        r0_py = np.loadtxt('data/rdist0.txt')
        r1_py = np.loadtxt('data/rdist1.txt')
        r2_py = np.loadtxt('data/rdist2.txt')

        self.assertTrue(np.allclose(rd0,r0_py))
        self.assertTrue(np.allclose(rd1,r1_py))
        self.assertTrue(np.allclose(rd2,r2_py))
    """   
    def test_vacuum_form_factor_data(self):
        print "Testing vacuum form factor data against libsaxs...\n"
        vls = np.loadtxt('../data/vacuum_ls.txt')
        
        vpy = np.loadtxt('../data/vacuum_ff.txt')

        self.assertTrue(np.allclose(vls,vpy))

    def test_dummy_form_factor_data(self):
        print "Testing dummy form factor data against libsaxs...\n"
        dpy = np.loadtxt('../data/dummy_ff.txt')
        dls = np.loadtxt('../data/dummy_ls.txt')

        self.assertTrue(np.allclose(dpy,dls))

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
        q = np.loadtxt('../data/q_impPy.txt')
        self.assertTrue(np.allclose(qls,q))

    def test_Gq_data(self):
        print "Testing Gq data against libsaxs...\n"
        gqls = np.loadtxt('data/gq_ls.txt')
        gq = np.loadtxt('data/gq_impPy.txt')
        self.assertTrue(np.allclose(gqls,gq))

    def test_intensity_data(self):
        print "Teesting intensity data against libsaxs...\n"
        ils = np.loadtxt('data/intensity_ls.txt')
        intens = np.loadtxt('data/intensity_impPy.txt')

        self.assertTrue(np.allclose(ils,intens))
        
if __name__ == '__main__':
    unittest.main()
