import unittest
import sys
import pysingfel.solvent_form_factor as sff


class FormFactorTest(unittest.TestCase):
        
    def setUp(self):
        self.ft = sff.FormFactorTable()
        
    def test_read_atomic_coefficients(self):
    
        #fft = form_factor_table.FormFactorTable('formfactors-int_tab_solvation.lib',0,3.0,0.01)
        self.assertIsInstance(self.ft,sff.FormFactorTable)
        
    def test_vanderwaals_radius(self):
        
        self.assertEqual(self.ft.get_vanderwaals_radius(2),1.82)
        self.assertEqual(self.ft.get_vanderwaals_radius(-3),1.75)
        
    def test_element_form_factor_dict(self):
        self.assertEqual(self.ft.get_element_dict_id(0),'H')
        self.assertEqual(self.ft.get_element_dict_id(5),'C')
        self.assertEqual(self.ft.get_element_dict_id(28),'Ag')
      
if __name__ == '__main__':
    unittest.main()
        

