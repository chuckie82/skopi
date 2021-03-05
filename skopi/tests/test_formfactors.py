from skopi.ff_waaskirf_database import load_waaskirf_database
import numpy as np
import skopi as sk

def test_form_factor():
    """
    Test form factor calculation at a random radial position in q-space.
    """
    
    # get random value of sin(theta)/labmda (stol) in range(0,0.6) where ffs are defined 
    # note that stol = np.linalg.norm(q_vector) / (4*np.pi)
    stol =  np.random.uniform(0,high=0.6)
    
    # generate mock particle
    particle = sk.Particle()
    particle.create_from_atoms([('O',np.random.randn(3)),
                                ('C',np.random.randn(3)),
                                ('N',np.random.randn(3))])
    
    # compute form factors using diffraction.py method
    wk_dbase = load_waaskirf_database()
    form_factor = sk.diffraction.calculate_atomic_factor(particle, stol, 1)
    
    # list from https://bruceravel.github.io/demeter/pods/Xray/Scattering/WaasKirf.pm.html
    # tables can be found at Waasmaier and Kifel. Acta Cryst (1995): A51.
    waaskirf_atom_list = [ 'H', 'H1-', 'He', 'Li',  'Li1+', 'Be', 'Be2+', 'B', 'C', 'Cval' ,'N' ,'O']

    # check against alternative approach
    for i,atom_type in enumerate(['O','C','N']):
        index = waaskirf_atom_list.index(atom_type)
        a_coeffs = wk_dbase[index][2:2+5]
        b_coeffs = wk_dbase[index][8:8+5]
        c_coeff = wk_dbase[index][7]
        ff_val = np.sum(a_coeffs * np.exp(-1*b_coeffs * np.square(stol))) + c_coeff
        assert np.allclose(ff_val, form_factor[i])
        
    return
