import sys,os
import numpy as np
sys.path.append("../..")
import skopi as sk
import scipy
import argparse
import matplotlib.pyplot as plt
import time


def main():
    

    params = parse_input_arguments(sys.argv)
    min_q = params['min_q']
    max_q = params['max_q']
    delta_q = params['delta_q']
    pdb = params['pdb']
    c1 = params['c1']
    c2 = params['c2']
    ff_table_file = params['form_factor_table']

  
    
    particles = sk.particle.Particle()
    particles.read_pdb(pdb,'CM')

    elements = np.unique(particles.get_atom_type()).astype(np.int32)
    q_entries = particles.get_q_sample() 
    
    elements = elements.reshape(len(elements),1)


    excl_vol = np.zeros((len(elements),1),dtype=np.float64)

    ft = sk.solvent_form_factor.form_factor_table.FormFactorTable(ff_table_file,min_q,max_q,delta_q)


    

    vacuum_ff = ft.get_vacuum_form_factors()
    dummy_ff = ft.get_dummy_form_factors()

    radius = []

    xyz = particles.get_atom_struct()
    xyz = np.transpose(xyz)


     
    atomic_variant = particles.get_atomic_variant()
    symbols = particles.get_atomic_symbol()
    residue  = particles.get_residue()
   
    table = ft.get_ff_cm_dict()
    for m in range(len(xyz)):
           if m % 1000 == 0:
              print m

           ret_type = ft.get_form_factor_atom_type(symbols[m],atomic_variant[m], residue[m])
           print ret_type
           idx =   table[ret_type]

           radius.append(ft.ff_radii[idx])

    radius = np.asarray(radius)
    radius = radius.reshape((len(xyz),1))
    print radius.shape
    radius = np.array(radius)
    print radius
    
    xyz_plus_radius = np.hstack((xyz,radius))
    start = time.time()


    s = sk.solvent_form_factor.solvent_accessible_surface.SolventAccessibleSurface()
    surface_area,fraction,sas = s.calculate_asa(xyz_plus_radius,1.4,960)
    end = time.time()
    print 'Calculated %d particle surface areas in %f seconds.' % (len(xyz_plus_radius),end-start)
    
    np.savetxt("sas.txt",fraction) 
    start = time.time()
    model_profile = sk.solvent_form_factor.saxs_profile.Profile(min_q,max_q,delta_q)

    intensity  = sk.solvent_form_factor.saxs_profile.calculate_profile_partial(model_profile,particles,fraction,ft,vacuum_ff,dummy_ff,c1,c2)
    model_profile.write_SAXS_file('SAXS_intensities.txt')
    model_profile.write_partial_profiles('SAXS_partial_profiles.txt')

    end = time.time()
    print 'Calculated %d particle profiles in %f seconds.' % (len(fraction),end-start)



def parse_input_arguments(args):

    del args[0]
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", type=str,help="PDB filename")
    parser.add_argument("--c1", default = 1.0, type=float,help="excluded volume SAXS (c1) parameter")
    parser.add_argument("--c2", default = 0.0, type=float, help="hydration layer SAXS (c2) parameter")
    parser.add_argument("--min_q", default=0.0, type=float, help="minimum q value")
    parser.add_argument("--max_q", default=1.0, type=float, help="maximum q value")
    parser.add_argument("--delta_q", default=0.01, type=float, help="delta q value")
    parser.add_argument("--form_factor_table",default=None, type=str, help="form_factor_table_file")
    

    

    return vars(parser.parse_args(args))


if __name__ == '__main__':

   main()
