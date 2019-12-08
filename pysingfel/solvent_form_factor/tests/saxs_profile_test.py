
import unittest
import sys,os
import numpy as np
sys.path.append('../../..')
import pysingfel as ps

class TestPartialProfiles(unittest.TestCase):
    
    def setUp(self):
        
        self.verbose=1 
        self.ft = ps.solvent_form_factor.form_factor_table.FormFactorTable()
        self.prof = ps.solvent_form_factor.saxs_profile.Profile(0,3.0,0.01)
        self.particle = ps.particle.Particle()
        self.particle.read_pdb('SAXS_10atoms.pdb','CM')
        self.nsamples = 301
        self.coordinates = self.particle.get_atom_pos()
        
        self.atom_type = self.particle.get_atom_type().astype(int)
        
        self.vac = np.zeros((len(self.coordinates),self.nsamples),dtype=np.float64)
        self.dum = np.zeros((len(self.coordinates),self.nsamples),dtype=np.float64)
        
        self.vv = self.ft.get_vacuum_form_factors()
        self.dd = self.ft.get_dummy_form_factors()
       
        self.saxs_sa = np.zeros((len(self.coordinates),1),dtype=np.float64)
        self.max_dist = ps.solvent_form_factor.saxs_profile.calculate_max_distance(self.coordinates)
        
        self.r_dist = []
        self.ndists = 6
        for i in range(self.ndists):
           self.r_dist.append(ps.solvent_form_factor.radial_distribution_function.RadialDistributionFunction(0.5,self.max_dist))
           
        for i in range(len(self.coordinates)):
            self.vac[i,0] = self.ft.get_vacuum_zero_form_factors(self.atom_type[i-1])
            self.dum[i,0] = self.ft.get_dummy_zero_form_factors(self.atom_type[i-1])
 
        
    def test_assign_form_factors(self):
    
        particles = ps.particle.Particle()
        particles.read_pdb('SAXS_10atoms.pdb','CM')
        lp = particles.get_num_atoms()
        ft_py = ps.solvent_form_factor.form_factor_table.FormFactorTable()
        symbols = particles.get_atomic_symbol()
        atomic_variant = particles.get_atomic_variant()
        residue  = particles.get_residue()
        table = ft_py.get_ff_cm_dict()
        vff_py=[]
        dff_py=[]
        vacuum_ff_m  = ft_py.get_vacuum_form_factors()
        dummy_ff_m  = ft_py.get_dummy_form_factors()
        vacuum_ff_m[:,-1] = 0.0
        dummy_ff_m[:,-1] = 0.0
        for i in range(lp):
        
            ret_type = ft_py.get_form_factor_atom_type(symbols[i],atomic_variant[i], residue[i])
        
            idx = table[ret_type]
            vff_py.append(vacuum_ff_m[idx])
            dff_py.append(dummy_ff_m[idx])
        mv = np.asarray(vff_py)
        md = np.asarray(dff_py)

    def test_check_water_form_factor(self):
    
       
       self.assertEqual(self.ft.get_water_form_factor(),3.50968)
       
    def test_assign_form_factors_2_profile(self):
        print "Here is test for assign_form_factors_to_profile\n"
    
    def test_calculate_partial_profiles(self):
        print "Here is test for calculate_partial_profiles\n"

    def test_calculate_max_distance(self):
        
        coord1 = [1,2,3]
        coord2 = [6,5,6]
        coord3 = [9,1,1]
        coord = np.array([coord1,coord2,coord3])
        max_d = ps.solvent_form_factor.saxs_profile.calculate_max_distance(coord)

        self.assertEqual(max_d,np.sqrt(69))
    
        
    def test_check_radial_distribution(self):
        num_atoms = len(self.coordinates)
        p,waterff,ndists= ps.solvent_form_factor.saxs_profile.assign_form_factors_2_profile(self.particle,self.prof,self.saxs_sa,self.vv,self.dd,self.ft,num_atoms,self.ndists,self.verbose)
        
        f = np.zeros((6,1),dtype=np.float64)
        
        for i in range(len(self.coordinates)):
            
            for j in range(i+1,len(self.coordinates)):
    
                f[0] += 2.0*p.vacuum_ff[i,0]*p.vacuum_ff[j,0]
                f[1] += 2.0*p.dummy_ff[i,0]*p.dummy_ff[j,0]
                f[2] += 2.0*(p.vacuum_ff[i,0] * p.dummy_ff[j,0] +
                p.vacuum_ff[j,0] * p.dummy_ff[i,0])
                f[3] += 0  # values don't matter, only profile and rdf check
                f[4] += 0
                f[5] += 0
                
        r0  = float(self.r_dist[0].get_values())
        print r0
        r1 = float(self.r_dist[1].get_values())
        r2 = float(self.r_dist[2].get_values())
        r3 = float(self.r_dist[3].get_values())
        r4 = float(self.r_dist[4].get_values())
        r5 = float(self.r_dist[5].get_values())
        r = np.array([r0, r1 ,r2,r3,r4,r5])
        
        np.allclose(r,f)
        
    
    def test_radial_distribution_to_partial_profiles(self):
        num_atoms = len(self.coordinates)  
        r_dist = []
        ndists = 6
        self.bin_size = 0.5
        for i in range(ndists):
           r_dist.append(ps.solvent_form_factor.radial_distribution_function.RadialDistributionFunction( self.bin_size,self.max_dist))
        p,waterff, ndists = ps.solvent_form_factor.saxs_profile.assign_form_factors_2_profile(self.particle,self.prof,self.saxs_sa ,self.vv,self.dd,self.ft,num_atoms,ndists,self.verbose)
        
        new_p = ps.solvent_form_factor.radial_distribution_function.radial_distributions_to_partials(p,6,r_dist,self.verbose)
 
        self.assertIsInstance(new_p,ps.solvent_form_factor.saxs_profile.Profile)
        
        self.assertFalse(np.any(new_p.vac_vac),0.0)
        self.assertFalse(np.any(new_p.dum_dum),0.0)
        self.assertFalse(np.any(new_p.vac_dum),0.0)
        self.assertFalse(np.any(new_p.h2o_h2o),0.0)
        self.assertFalse(np.any(new_p.dum_h2o),0.0)
        self.assertFalse(np.any(new_p.vac_h2o),0.0)
        

    def test_sum_partial_profiles(self):
        z = np.zeros((self.nsamples,1),dtype=np.float64)
        ndists =6
        num_atoms = len(self.coordinates)
        p,waterff,r_size  = ps.solvent_form_factor.saxs_profile.assign_form_factors_2_profile(self.particle,self.prof,self.saxs_sa,self.vv,self.dd,self.ft,num_atoms,ndists,self.verbose)
        self.assertIsInstance(p,ps.solvent_form_factor.saxs_profile.Profile)
        
        new_p = ps.solvent_form_factor.radial_distribution_function.radial_distributions_to_partials(p,ndists,self.r_dist,self.verbose)
        
        self.assertEqual(np.any(new_p.vac_vac),0.0)
        self.assertEqual(np.any(new_p.dum_dum),0.0)
        self.assertEqual(np.any(new_p.vac_dum),0.0)
        self.assertEqual(np.any(new_p.h2o_h2o),0.0)
        self.assertEqual(np.any(new_p.dum_h2o),0.0)
        self.assertEqual(np.any(new_p.vac_h2o),0.0)
        
        Intensity = ps.solvent_form_factor.saxs_profile.sum_profile_partials(new_p,1.0,0.0,self.verbose)
        self.assertTrue(np.allclose(Intensity,z))
    

if __name__ == '__main__':
    unittest.main()
