import unittest
import form_factor_table
import radial_distribution_function
import saxs_profile
import sys,os
import particle
import numpy as np

class TestRadialDistributionFunction(unittest.TestCase):

    def setUp(self):
        self.max_dist = 10
        self.bin_size = 0.5

        self.rdf = radial_distribution_function.RadialDistributionFunction(self.bin_size,self.max_dist)
        self.nbins = self.rdf.get_nbins()
        
    def test_rdf_object(self):
    
        self.assertIsInstance(self.rdf,radial_distribution_function.RadialDistributionFunction)
        
    def test_rdf_nbins(self):
    
        self.assertEqual(self.rdf.get_nbins(),21)
        
    def test_rdf_max_distance(self):
    
        self.assertEqual(self.max_dist,self.rdf.get_max_distance())
    
    def test_rdf_reset(self):
        
        r = np.zeros((self.nbins),dtype=np.float64)
        self.rdf.reset()
        v = self.rdf.get_values()
        np.allclose(v,r)
        
    def test_values(self):

        z = np.zeros((self.nbins,1),dtype=np.float64)
        radial_distribution_function.add2distribution(self.rdf,self.max_dist,20) # put 20 in last bin
        z[-1] = 20
        np.allclose(z,self.rdf.values)
            
if __name__ == '__main__':
    unittest.main()

