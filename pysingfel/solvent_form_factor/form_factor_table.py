
# form_factor_table.py

from collections import defaultdict
import numpy as np
import sys
sys.path.append('../..')
import pysingfel as ps

"""

class FormFactorTable:

   class that deals with form factor computation
   
   Each form factor can be divided into two parts: vacuum and dummy.
   dummy is an approximated excluded volume (solvent) form factor.
   The approximation is done using Fraser, MacRae and Suzuki (1978) model.
   
"""


class FormFactorTable(object):
    _CARBON_ATOMS_DICT = {
        'CH': defaultdict(lambda: 'CH'),
        'CH2': defaultdict(lambda: 'CH2'),
        'CH3': defaultdict(lambda: 'CH3'),
        'C': defaultdict(lambda: 'C'),
        'CA': defaultdict(lambda: 'CH', {'GLY': 'CH2'}),  # Gly has 2 H
        'CB': defaultdict(lambda: 'CH2', {'ILE': 'CH','THR': 'CH','VAL': 'CH','ALA': 'CH3'}),
        'CG': defaultdict(lambda: 'CH2', {'ASN': 'C','ASP': 'C','HIS': 'C','PHE': 'C','TRP': 'C','TYR': 'C','LEU': 'CH'}),
        'CG1': {'ILE': 'CH2', 'VAL': 'CH3'},  # No default
        'CG2': defaultdict(lambda: 'CH3'),
        'CD': defaultdict(lambda: 'CH2', {'GLU': 'C','GLN': 'C'}),
        'CD1': defaultdict(lambda: 'C', {'LEU': 'CH3', 'ILE': 'CH3', 'PHE': 'CH', 'TRP': 'CH', 'TYR': 'CH'}),
        'CD2': defaultdict(lambda: 'C', {'LEU': 'CH3','PHE': 'CH','HIS': 'CH', 'TYR': 'CH'}),
        'CE': defaultdict(lambda: 'C', {'LYS': 'CH2','MET': 'CH3'}),
        'CE1': defaultdict(lambda: 'C', {'PHE': 'CH','HIS': 'CH','TYR': 'CH'}),
        'CE2': defaultdict(lambda: 'C', {'PHE': 'CH','TYR': 'CH'}),
        'CZ': defaultdict(lambda: 'C', {'PHE': 'CH'}),
        'CZ1': defaultdict(lambda: 'C'),
        'CZ2': defaultdict(lambda: 'C', {'TRP': 'CH'}),
        'CZ3': defaultdict(lambda: 'C', {'TRP': 'CH'}),
        'CE3': defaultdict(lambda: 'C', {'TRP': 'CH'}),
        'C1p': defaultdict(lambda: 'CH'),
        'C2p': defaultdict(lambda: 'CH'),
        'C3p': defaultdict(lambda: 'CH'),
        'C4p': defaultdict(lambda: 'CH'),
        'C5p': defaultdict(lambda: 'CH2'),
        'C2': defaultdict(lambda: 'C', {'DADE': 'CH','ADE': 'CH'}),
        'C4': defaultdict(lambda: 'C'),
        'C5': defaultdict(lambda: 'C', {'DCYT': 'CH','CYT': 'CH','DURA': 'CH','URA': 'CH'}),
        'C6': defaultdict(lambda: 'C', {'DCYT': 'CH','CYT': 'CH','DURA': 'CH','URA': 'CH','DTHY': 'CH','THY': 'CH'}),
        'C7': defaultdict(lambda: 'CH3'),
        'C8': defaultdict(lambda: 'CH'),
    }

    _NITROGEN_ATOMS_DICT = {
        'N': defaultdict(lambda: 'NH', {'PRO': 'N'}),
        'ND': defaultdict(lambda: 'N'),
        'ND1': defaultdict(lambda: 'N', {'HIS':  'NH'}),
        'ND2': defaultdict(lambda: 'N', {'ASN': 'NH2'}),
        'NH1': defaultdict(lambda: 'N', {'ARG': 'NH2'}),
        'NH2': defaultdict(lambda: 'N', {'ARG': 'NH2'}),
        'NE': defaultdict(lambda: 'N', {'ARG': 'NH'}),
        'NE1': defaultdict(lambda: 'N', {'TRP': 'NH'}),
        'NE2': defaultdict(lambda: 'N', {'GLN': 'NH2'}),
        'NZ': defaultdict(lambda: 'N', {'LYS': 'NH3'}),
        'N1': defaultdict(lambda: 'N', {'DGUA': 'NH', 'GUA': 'NH'}),
        'N2': defaultdict(lambda: 'NH2'),
        'N4': defaultdict(lambda: 'NH2'),
        'N6': defaultdict(lambda: 'NH2'),
        'N3': defaultdict(lambda: 'N', {'DURA': 'NH', 'URA': 'NH'}),
        'N7': defaultdict(lambda: 'N'),
        'N9': defaultdict(lambda: 'N'),
    }

    # table name currently not used
    def __init__(self,table_name=None,mnq=0.0,mxq=3.0,dq=0.01):
    
        self.max_q_ = mxq
        self.min_q_ = mnq
        self.delta_q_ = dq
                        
                         
        # Elements for Cromer-Mann dictionary
        self.ff_cm_dict = { 'H':0,'HE':1,'C':2,'N':3,'O':4,'NE':5,'SOD+':6,'MG+2':7,'P':8,'S':9,'CAL2+':10,'FE2+':11,'ZN2+':12,'SE':13,'AU':14,'CH':15,'CH2':16,'CH3':17,'NH':18,'NH2':19,'NH3':20,'OH':21,'OH2':22,'SH':23}
        

        
        #                i=1,5
        #  a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5  excl_vol
        #H He C N O Ne Na Mg Cl
        # Cromer-Mann coefficients
        
        self.ff_coeff      = np.array([[0.493002  , 0.322912  , 0.140191  , 0.040810 ,  0.0  ,   0.003038 ,  10.5109  , 26.1257   ,  3.14236 ,  57.7997   ,  1.0,      5.15],

        [0.489918  ,  0.262003 ,  0.196767  , 0.049879  , 0.0  ,   0.001305 ,  20.6593  ,  7.74039 ,  49.5519   ,  2.20159   , 1.0    ,  5.15],

        [2.31000  ,  1.02000 ,   1.58860 ,   0.865000  , 0.0   ,  0.215600   ,20.8439  , 10.2075  ,   0.568700  ,  51.6512  ,   1.0    ,  16.44],

        [12.2126   ,  3.13220  ,  2.01250  ,  1.16630 ,   0.0   , -11.529   ,    0.005700 , 9.89330  , 28.9975  ,   0.582600  , 1.0   ,   2.49],
        [3.04850 ,   2.28680 ,   1.54630 ,   0.867000 ,  0.0  ,   0.250800 ,   13.2771,     5.70110  ,  0.323900  ,32.9089  ,  1.0    ,  9.13],
        [3.95530  ,  3.11250 ,   1.45460 ,    1.12510  ,  0.0 ,    0.351500 ,    8.40420 ,  3.42620  ,  0.230600 , 21.7184 ,    1.0     , 9.],

        [4.76260  ,  3.17360    ,1.26740   , 1.11280   , 0.0  ,  0.676000 ,   3.28500  , 8.84220  ,  0.313600 , 129.424   ,   1.0   ,   9. ],
        [5.42040 ,   2.17350  ,  1.22690  ,  2.30730  ,  0.0  ,   0.858400  ,  2.82750 , 79.2611   ,  0.380800  , 7.19370  ,  1.0   ,   9.],
        [6.43450   , 4.17910 ,   1.78000   , 1.49080  ,  0.0    , 1.11490   ,  1.90670 , 27.1570   ,  0.526000 , 68.1645   ,  1.0    ,  5.73],
        [6.90530   , 5.20340   , 1.43790   , 1.58630  ,  0.0    , 0.866900   , 1.46790 , 22.2151 ,    0.253600  ,56.1720  ,   1.0   ,   19.86],
        #[18.2915 ,    7.20840  ,  6.53370  ,  2.33860   , 0.0  , -16.378    ,   0.00660  , 1.1717  ,  19.5424  ,  60.4486  ,   1.0  ,    9.],
        [15.6348   , 7.95180  ,  8.43720   , 0.853700 ,  0.0  , -14.875   ,   -0.00740  , 0.608900 , 10.3116  ,  25.9905   ,  1.0   ,   9.],
        [11.0424   ,  7.37400  ,  4.13460 ,   0.439900  , 0.0   ,  1.00970  ,   4.65380  , 0.305300 , 12.0546   , 31.2809  ,   1.0   ,   9.],
        #[11.2296  ,   7.38830 ,  4.73930 ,   0.71080  ,  0.0  ,   0.93240 ,    4.12310  , 0.272600  ,10.2443  ,  25.6466   ,  1.0    ,  9.],
        [11.9719   ,  7.38620  ,  6.46680  ,  1.39400  ,  0.0   ,  0.780700 ,   2.99460 ,  0.203100 ,  7.08260  , 18.0995   ,  1.0    ,  9.],
        [17.0006 ,    5.81960 ,    3.97310   , 4.35430  ,  0.0  ,   2.84090  ,   2.40980  , 0.272600  , 15.2372  , 43.8163  ,   1.0   ,   9.],
        [16.8819   , 18.5913  ,  25.5582  ,   5.86000   , 0.0  ,   12.0658   ,  0.461100 , 8.62160    , 1.48260  ,36.3956   ,  1.0    ,  19.86]])
        
        self.vol_coefficient = np.zeros((self.ff_coeff.shape[0],1),dtype=np.float64)
        
        self.a = np.zeros((self.ff_coeff.shape[0],5),dtype=np.float64)
        self.b = np.zeros((self.ff_coeff.shape[0],5),dtype=np.float64)
        self.c = np.zeros((self.ff_coeff.shape[0],1),dtype=np.float64)
        self.excl_vol = np.zeros((self.ff_coeff.shape[0],1),dtype=np.float64)
        
        for i in range(self.ff_coeff.shape[0]):
            for j in range(5):
            
               self.a[i,j]   =   self.ff_coeff[i,j]
               
            self.c[i] = self.ff_coeff[i,5]
            
            for j in range(5):
            
                self.b[i,j-5] = self.ff_coeff[i,j+6]
                
            self.excl_vol[i] = self.ff_coeff[i,11]
        
      
        self.number_of_q_entries = int(np.ceil((self.max_q_ - self.min_q_) / self.delta_q_))+1
        self.q_ =np.zeros((self.number_of_q_entries,1),dtype=np.float64)
        
        self.vanderwaals_volume = None

        self.form_factor_table = None

        self.rho = 0.334 # electron density of water
        self.water_form_factor = 3.50968

        self.form_factor_coefficients = None
        self.ff_table = None

        self.vacuum_form_factors = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)
        self.dummy_form_factors = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)
        self.ff_radii = np.zeros((self.ff_coeff.shape[0],1),dtype=np.float64)       


        self.compute_form_factors_all_atoms()
        self.compute_dummy_form_factors()
        self.compute_form_factors_heavy_atoms()
        self.compute_radii()


   

    def get_volume(self,p):

        form_factor = self.get_dummy_form_factors(p)

        return form_factor/self.rho


    

    def get_dummy_form_factors(self):

        return self.dummy_form_factors

    def set_dummy_form_factors(self, dum):

        self.dummy_form_factors = dum
        
    def get_vacuum_form_factors(self):

        return self.vacuum_form_factors
    
    def set_vacuum_form_factors(self,v):
        
        self.vacuum_form_factors = v

    def get_default_form_factor_table(self):

        return self.form_factor_table

    # Cromer-Mann form factor dictionary
    def get_ff_cm_dict(self):
        
        return self.ff_cm_dict
    
    def get_form_factor_table(self):

        return self.form_factor_table

    def get_vacuum_form_factors(self):

        return self.vacuum_form_factors

    def set_vacuum_form_factors(self,fft):

        self.vacuum_form_factors = fft

    def get_dummy_form_factors(self):

        return self.dummy_form_factors

    def set_dummy_form_factors(self,dum):

        self.dummy_form_factors = dum

        
    def get_water_form_factor(self):
        
        return self.water_form_factor


    def compute_dummy_form_factors(self):

  
        dff = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)

        for i in range(self.ff_coeff.shape[0]):
           
            for iq in range(self.number_of_q_entries):

                dff[i,iq] = self.rho * self.excl_vol[i] * np.exp(-np.power(self.excl_vol[i],(2.0/3.0)) * (self.q_[iq]**2)/(16.0*np.pi))
                
                
        self.dummy_form_factors = dff
    
   
    def compute_radii(self):
        
        one_third = (1.0/3.0)
        c = (3.0/(4.0*np.pi))

        for i in range(0,self.ff_coeff.shape[0]):

            form_factor = self.get_dummy_form_factors()
            ff = form_factor[:,0]/self.rho
        
        self.ff_radii = np.power(c*ff, one_third)

    
    
            
    def compute_form_factors_all_atoms(self):
        
       
        s = np.zeros((self.number_of_q_entries,1),dtype=np.float64)
        for i in range(0,self.ff_coeff.shape[0]):


            
            for iq in range(self.number_of_q_entries):

                self.q_[iq] = self.min_q_  + float(iq)*self.delta_q_
                s[iq] = self.q_[iq]/(4.0*np.pi)
                
                

                self.vacuum_form_factors[i,iq] = self.c[i] 

                for j in range(5):

                    self.vacuum_form_factors[i,iq] += self.a[i,j] * np.exp(-self.b[i,j]*s[iq]*s[iq])


        
            

# end of function: compute_form_factors_all_atoms
       
    
    def compute_form_factors_heavy_atoms(self):
        
        """
        Calculating the form factors of the heavy atoms
        
        """
        vacuum_form_factors_ch =  self.vacuum_form_factors[2] + 1.0*self.vacuum_form_factors[0]
        vacuum_form_factors_ch2 =  self.vacuum_form_factors[2] + 2.0 * self.vacuum_form_factors[0]
        vacuum_form_factors_ch3 =  self.vacuum_form_factors[2] + 3.0 * self.vacuum_form_factors[0]
        dummy_form_factors_ch =  self.dummy_form_factors[2] + 1.0*self.dummy_form_factors[0]
        dummy_form_factors_ch2 =  self.dummy_form_factors[2] + 2.0 * self.dummy_form_factors[0]
        dummy_form_factors_ch3 =  self.dummy_form_factors[2] + 3.0 * self.dummy_form_factors[0]
        
        vacuum_form_factors_nh =  self.vacuum_form_factors[3] + 1.0*self.vacuum_form_factors[0]
        vacuum_form_factors_nh2 =  self.vacuum_form_factors[3] + 2.0 * self.vacuum_form_factors[0]
        vacuum_form_factors_nh3 =  self.vacuum_form_factors[3] + 3.0 * self.vacuum_form_factors[0]
        dummy_form_factors_nh =  self.dummy_form_factors[3] + 1.0*self.dummy_form_factors[0]
        dummy_form_factors_nh2 =  self.dummy_form_factors[3] + 2.0 * self.dummy_form_factors[0]
        dummy_form_factors_nh3 =  self.dummy_form_factors[3] + 3.0 * self.dummy_form_factors[0]
        
        vacuum_form_factors_oh =  self.vacuum_form_factors[4] + 1.0*self.vacuum_form_factors[0]
        vacuum_form_factors_oh2 =  self.vacuum_form_factors[4] + 2.0 * self.vacuum_form_factors[0]
        dummy_form_factors_oh =  self.dummy_form_factors[4] + 1.0*self.dummy_form_factors[0]
        dummy_form_factors_oh2 =  self.dummy_form_factors[4] + 2.0 * self.dummy_form_factors[0]
        
        vacuum_form_factors_sh =  self.vacuum_form_factors[9] + 1.0*self.vacuum_form_factors[0]
        dummy_form_factors_sh =  self.dummy_form_factors[9] + 1.0 * self.dummy_form_factors[0]
        
        heavy_vacuum_form_factors = [vacuum_form_factors_ch,vacuum_form_factors_ch2,vacuum_form_factors_ch3,
            vacuum_form_factors_nh,vacuum_form_factors_nh2,vacuum_form_factors_nh3,
            vacuum_form_factors_oh,vacuum_form_factors_oh2,vacuum_form_factors_sh]
        heavy_dummy_form_factors = [dummy_form_factors_ch,dummy_form_factors_ch2,dummy_form_factors_ch3,
            dummy_form_factors_nh,dummy_form_factors_nh2,dummy_form_factors_nh3,
            dummy_form_factors_oh,dummy_form_factors_oh2,dummy_form_factors_sh]
        
        self.vacuum_form_factors = np.vstack((self.vacuum_form_factors,heavy_vacuum_form_factors))
        self.dummy_form_factors = np.vstack((self.dummy_form_factors,heavy_dummy_form_factors))
       
        

    def get_carbon_atom_type(self, atomic_variant_type,residue_type):
        """
         Determining the carbon complex for the form factor
        :param atomic_variant_type: atomic variant of carbon
        :param residue_type: the resdiue that contains the carbon
        :return carbon complex returned as string constant
        """
        try:
          return self._CARBON_ATOMS_DICT[atomic_variant_type][residue_type]
        except KeyError:
          print("Carbon atom not found, using default C form factor for"
                "atomic_variant=%s, residue=%s"
                "" % (atomic_variant_type,residue_type))
          return 'C'


    def get_nitrogen_atom_type(self, atomic_variant_type, residue_type):
        """
        Determining the nitrogen complex for the form factor
        :param atomic_variant_type: atomic variant of nitrogen
        :param residue_type: the residue that contains the nitrogen
        :return nitrogen complex returned as string constant

        """
        try:
          return self._NITROGEN_ATOMS_DICT[atomic_variant_type][residue_type]
        except KeyError:
          print("Nitrogen atom not found, using default N form factor for"
                "atomic_variant=%s, residue=%s"
                "" % (atomic_variant_type,residue_type))
          return 'N'

    def get_sulfur_atom_type(self,atomic_variant_type,residue_type):

        """
         Determining the sulfur complex for the form factor
        :param atomic_variant_type: atomic variant of sulfur
        :param residue_type: the residue that contains the sulfur
        :return sulfur complex returned as string constant

        """
        # SD
        if atomic_variant_type == 'SD':
           return 'S'
        #SG
        if atomic_variant_type == 'SG':
           if residue_type == 'CYS':
              return 'SH'
           return 'S'
            
        print("Sulfur atom not found, using default S form factor for atomic_variant=%s, residue=%s\n" % (atomic_variant_type,residue_type))
        return 'S'
        

    def get_oxygen_atom_type(self,atomic_variant_type,residue_type):

        """
        Determining the oxygen complex for the form factor
        :param atomic_variant_type: atomic variant of oxygen
        :param residue_type: the residue that contains the oxygen
        :return oxygen complex returned as string constant

        """
 
        # O OE1 OE2 OD1 OD2 O1A O2A OXT OT1 OT2
        if atomic_variant_type == 'O' or atomic_variant_type == 'OE1' or atomic_variant_type == 'OE2' or atomic_variant_type == 'OD1' or atomic_variant_type == 'OD2' or atomic_variant_type == 'O1A' or atomic_variant_type == 'O2A' or atomic_variant_type == 'OT1' or atomic_variant_type == 'OT2' or atomic_variant_type == 'OXT':
           return 'O'
   
        # OG
        if atomic_variant_type == 'OG':
            if residue_type == 'SER':
                return 'OH'
            return 'O'
   
        # OG1
        if atomic_variant_type == 'OG1':
            if residue_type == 'THR':
                return 'OH'
            return 'O'
   
        # OH
        if atomic_variant_type == 'OH':
            if residue_type == 'TYR':
                return 'OH'
            return 'O'
   
        # DNA/RNA atoms
        # O1P, O3', O2P, O2',O4',05', O2,O4,O6
        if atomic_variant_type == 'OP1' or atomic_variant_type == 'O3p' or atomic_variant_type == 'OP2' or atomic_variant_type == 'O2p' or atomic_variant_type == 'O4p' or  atomic_variant_type == 'O5p' or atomic_variant_type == 'O2' or atomic_variant_type == 'O4' or atomic_variant_type == 'O6':
           return 'O'
   
        #  O2'
        if atomic_variant_type == 'O2p':
           return 'OH'

        # water molecule
        if residue_type == 'HOH':
           return 'OH2'

        print("Oxygen atom not found, using default O form factor for atomic_variant=%s, residue_type =%s\n" % (atomic_variant_type, residue_type))
                 
        return 'O'
 
    def get_form_factor_atom_type(self,atomic_type,atomic_variant_type, residue_type):
       """  
       Determining the form factor for an atom
       :param atomic_type: atomic element
       :param atomic_variant_type: type of element (C-alpha, C-beta, etc.)
       :param residue_type: residue that contains atom 
       :return ret_type: the complex used for the form factor of the atom
       """
       
       if atomic_type == 'C':
       
          ret_type = self.get_carbon_atom_type(atomic_variant_type, residue_type)

       elif atomic_type == 'N':
       
          ret_type = self.get_nitrogen_atom_type(atomic_variant_type, residue_type)
       elif atomic_type == 'O':
       
          ret_type = self.get_oxygen_atom_type(atomic_variant_type, residue_type)

       elif atomic_type == 'S':
       
          ret_type = self.get_sulfur_atom_type(atomic_variant_type, residue_type)

       # all other elements
       elif self.ff_cm_dict.has_key(atomic_type):
          ret_type = atomic_type

       # default N form factor if not found
       else:
          print("Can't find form factor for atom, using default value of nitrogen \n")
          ret_type = 'N'
       
       return ret_type;


    def get_radii(self,particles):

       """
       Gets the radii of the particle's atoms from the excluded volume (dummy form factor)
       :param  particles: the structure containing the atom info
       :return radii: the radii of all the atoms
       """

       num_atoms = particles.get_num_atoms()
       radii = np.zeros((num_atoms+1,),dtype=np.float64)
      
       symbols = particles.get_atomic_symbol()
       atomic_variant = particles.get_atomic_variant()
       residue = particles.get_residue()
           
       table = self.get_ff_cm_dict()
           
       for i in range(num_atoms+1):
    
           ret_type = self.get_form_factor_atom_type(symbols[i],atomic_variant[i], residue[i])

           idx = table[ret_type]
           radii[i] = self.ff_radii[idx]

       return radii
