import unittest
import sys
import pysingfel as ps
import pysingfel.solvent_form_factor as sff
import os
import numpy as np

class TestRadialDistributionFunction(unittest.TestCase):

    def setUp(self):
        self.max_dist = 10
        self.bin_size = 0.5

        self.rdf = sff.RadialDistributionFunction(self.bin_size,self.max_dist)
        self.nbins = self.rdf.get_nbins()
        
    def test_rdf_object(self):
    
        self.assertIsInstance(self.rdf,sff.RadialDistributionFunction)
        
    def test_rdf_nbins(self):
    
        self.assertEqual(self.rdf.get_nbins(),21)
        
    def test_rdf_max_distance(self):
    
        self.assertEqual(self.max_dist,self.rdf.get_max_distance())
    
    def test_rdf_reset(self):
        
        r = np.zeros((self.nbins),dtype=np.float64)
        self.rdf.reset()
        v = self.rdf.get_values()
        np.allclose(v,r)
    
    def test_radial_distribution_libsaxs_compare(self):
        print "test_radial_distribution goes here\n"

    def test_qd_var_ts(self):
        os.chdir('/reg/neh/home/marcgri/Software/pysingfel/pysingfel/solvent_form_factor/tests')
        pyQd10= np.loadtxt("data/qd_impPy.txt")
        lsQd10 = np.loadtxt("data/qd_ls.txt")
         
        self.assertTrue(np.allclose(pyQd10,lsQd10))

    def test_sinc_func(self):
        os.chdir('/reg/neh/home/marcgri/Software/pysingfel/pysingfel/solvent_form_factor/tests')

        
        pySinc10 = np.loadtxt("data/sinc_impPy.txt")
        lsSinc10 =  np.loadtxt("data/sinc_ls.txt")

        self.assertTrue(np.allclose(pySinc10,lsSinc10))

if __name__ == '__main__':
    unittest.main()

