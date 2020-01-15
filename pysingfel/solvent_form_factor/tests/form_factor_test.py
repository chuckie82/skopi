import unittest
import sys
sys.path.append('../../..')

import pysingfel as ps

class FormFactorTest(unittest.TestCase):
        
    def setUp(self):
        self.ft = ps.solvent_form_factor.form_factor_table.FormFactorTable()
        
    def test_read_atomic_coefficients(self):
    
        #fft = form_factor_table.FormFactorTable('formfactors-int_tab_solvation.lib',0,3.0,0.01)
        self.assertIsInstance(self.ft,ps.solvent_form_factor.form_factor_table.FormFactorTable)
        
    def test_vanderwaals_radius(self):
        
        self.assertEqual(self.ft.get_vanderwaals_radius(2),1.82)
        self.assertEqual(self.ft.get_vanderwaals_radius(-3),1.75)
        
      
if __name__ == '__main__':
    unittest.main()
        

