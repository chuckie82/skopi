import sys,os
import numpy as np
sys.path.append("../..")
import pysingfel as ps
import scipy
import argparse
import matplotlib.pyplot as plt
import time


def main():
    
    global verbose

    params = parse_input_arguments(sys.argv)
    min_q = params['min_q']
    max_q = params['max_q']
    delta_q = params['delta_q']
    pdb = params['pdb']
    c1 = params['c1']
    c2 = params['c2']
    verbose = params['verbose']
    ff_table_file = params['form_factor_table']

  
    
    particles = ps.particle.Particle()
    particles.read_pdb(pdb,'CM')
    lp = particles.get_num_atoms()

    elements = np.unique(particles.get_atom_type()).astype(np.int32)
    q_entries = particles.get_q_sample() 
    
    elements = elements.reshape(len(elements),1)


    excl_vol = np.zeros((len(elements),1),dtype=np.float64)

    ft = ps.solvent_form_factor.form_factor_table.FormFactorTable(ff_table_file,min_q,max_q,delta_q)


    for i in range(len(elements)):
    
       excl_vol[i] = ft.get_vanderwaals_volume(elements[i])

    vacuum_ff = ft.get_vacuum_form_factors()
    dummy_ff = ft.get_dummy_form_factors()

    radius = []

    xyz = particles.get_atom_struct()
    xyz = np.transpose(xyz)

    symbols = particles.get_atomic_symbol()
    e = particles.get_atomic_variant()

    residue = particles.get_residue()

    for i in range(len(xyz)):
        radius.append(2.0)
    radius = np.asarray(radius)
    radius = radius.reshape((lp,1))

    radius = np.array(radius)
    xyz_plus_radius = np.hstack((xyz,radius))
    start = time.time()


    s = ps.solvent_form_factor.solvent_accessible_surface.SolventAccessibleSurface()
    surface_area,fraction,sas = s.calculate_asa(xyz_plus_radius,1.4,100)
    end = time.time()
    print 'Calculated %d particle surface areas in %f seconds.' % (len(xyz_plus_radius),end-start)


    start = time.time()
    model_profile = ps.solvent_form_factor.saxs_profile.Profile(min_q,max_q,delta_q)

    intensity  = ps.solvent_form_factor.saxs_profile.calculate_profile_partial(model_profile,particles,fraction,ft,vacuum_ff,dummy_ff,verbose,c1,c2)
    model_profile.write_SAXS_file('SAXS_intensities.txt')
    model_profile.write_partial_profiles('SAXS_partial_profiles.txt')

    end = time.time()
    print 'Calculated %d particle profiles in %f seconds.' % (len(fraction),end-start)



def parse_input_arguments(args):

    del args[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=0, type=int, help="turn on  output verbosity")
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
