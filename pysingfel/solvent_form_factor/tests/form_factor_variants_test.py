import unittest
import form_factor_table

class FormFactorVariantsTest(unittest.TestCase):

    def setUp(self):
        self.ft = form_factor_table.FormFactorTable()
        self.atom_type = 'A'
        self.residue_type = 'XYZ'
    
    def test_carbon_atom_variants(self):
        
        self.assertEqual(self.ft.get_carbon_atom_type('C','XYZ'),'C')
        self.assertEqual(self.ft.get_carbon_atom_type('CA','ALA'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CA','GLY'),'CH2')
        
        rtb= ['ILE','THR','VAL']
        self.assertEqual(self.ft.get_carbon_atom_type('CB','ALA'),'CH3')
        for i in rtb:
            self.assertEqual(self.ft.get_carbon_atom_type('CB',i),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CB','GLY'),'CH2')
        
        rtg = ['ASN','ASP','HIS','PHE','TRP','TYR']
        for i in rtg:
            self.assertEqual(self.ft.get_carbon_atom_type('CG',i),'C')
        self.assertEqual(self.ft.get_carbon_atom_type('CG','LEU'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CG','XYZ'),'CH2')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CG1','ILE'),'CH2')
        self.assertEqual(self.ft.get_carbon_atom_type('CG1','VAL'),'CH3')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CG2','XYZ'),'CH3')
        self.assertEqual(self.ft.get_carbon_atom_type('CG2','ALA'),'CH3')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CD','GLN'),'C')
        self.assertEqual(self.ft.get_carbon_atom_type('CD','GLU'),'C')
        self.assertEqual(self.ft.get_carbon_atom_type('CD','ALA'),'CH2')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','LEU'),'CH3')
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','ILE'),'CH3')
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','PHE'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','TYR'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','TRP'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD1','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CD2','LEU'),'CH3')
        self.assertEqual(self.ft.get_carbon_atom_type('CD2','PHE'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD2','HIS'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD2','TYR'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CD2','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CE','LYS'),'CH2')
        self.assertEqual(self.ft.get_carbon_atom_type('CE','MET'),'CH3')
        self.assertEqual(self.ft.get_carbon_atom_type('CE','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CE1','PHE'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE1','HIS'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE1','TYR'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE1','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CE2','PHE'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE2','TYR'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE2','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CZ','PHE'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CZ','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CZ1','TYR'),'C')
        self.assertEqual(self.ft.get_carbon_atom_type('CZ1','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CZ2','TRP'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CZ2','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CZ3','TRP'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CZ3','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('CE3','TRP'),'CH')
        self.assertEqual(self.ft.get_carbon_atom_type('CE3','ALA'),'C')
        
        self.assertEqual(self.ft.get_carbon_atom_type('XYZ','XYZ'),'C')
        
    def test_nitrogen_atom_variants(self):
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('N','TRP'),'NH')
        self.assertEqual(self.ft.get_nitrogen_atom_type('N','ALA'),'NH')
        self.assertEqual(self.ft.get_nitrogen_atom_type('N','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND','ALA'),'N')
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND1','HIS'),'NH')
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND1','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND2','ASN'),'NH2')
        self.assertEqual(self.ft.get_nitrogen_atom_type('ND2','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('NH1','ARG'),'NH2')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NH1','PRO'),'N')
              
        self.assertEqual(self.ft.get_nitrogen_atom_type('NH2','ARG'),'NH2')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NH2','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE','ARG'),'NH')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE1','TRP'),'NH')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE1','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE2','GLN'),'NH2')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NE2','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('NZ','LYS'),'NH3')
        self.assertEqual(self.ft.get_nitrogen_atom_type('NZ','PRO'),'N')
        
        self.assertEqual(self.ft.get_nitrogen_atom_type('XYZ','XYZ'),'N')
        
    def test_oxygen_atom_variants(self):
    
        # O OE1 OE2 OD1 OD2 O1A O2A OXT OT1 OT2
        self.assertEqual(self.ft.get_oxygen_atom_type('O','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OE1','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OE2','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OD1','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OD2','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('O1A','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('O2A','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OXT','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OT1','XYZ'),'O')
        self.assertEqual(self.ft.get_oxygen_atom_type('OT2','XYZ'),'O')
        
        self.assertEqual(self.ft.get_oxygen_atom_type('OG','SER'),'OH')
        self.assertEqual(self.ft.get_oxygen_atom_type('OG','XYZ'),'O')
        
        self.assertEqual(self.ft.get_oxygen_atom_type('OG1','THR'),'OH')
        self.assertEqual(self.ft.get_oxygen_atom_type('OG1','XYZ'),'O')
        
        self.assertEqual(self.ft.get_oxygen_atom_type('OH','TYR'),'OH')
        self.assertEqual(self.ft.get_oxygen_atom_type('OH','XYZ'),'O')
        
        self.assertEqual(self.ft.get_oxygen_atom_type('XYZ','XYZ'),'O')
        
    def test_sulfur_atom_variants(self):
    
        self.assertEqual(self.ft.get_sulfur_atom_type('SD','TYR'),'S')
        self.assertEqual(self.ft.get_sulfur_atom_type('SD','XYZ'),'S')
        
        self.assertEqual(self.ft.get_sulfur_atom_type('SG','CYS'),'SH')
        self.assertEqual(self.ft.get_sulfur_atom_type('SG','XYZ'),'S')
        
        self.assertEqual(self.ft.get_sulfur_atom_type('XYZ','XYZ'),'S')
        
        
if __name__ == '__main__':
    unittest.main()
        
