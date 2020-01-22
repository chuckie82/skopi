import unittest
import sys
sys.path.append('../../..')

import pysingfel as ps

class FormFactorTest(unittest.TestCase):
        
    def setUp(self):
        self.ft = ps.solvent_form_factor.form_factor_table.FormFactorTable()
        
    def test_read_atomic_coefficients(self):
    
        self.assertIsInstance(self.ft,ps.solvent_form_factor.form_factor_table.FormFactorTable)
        
        
if __name__ == '__main__':
    unittest.main()
        

