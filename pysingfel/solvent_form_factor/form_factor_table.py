
# form_factor_table.py



import numpy as np
import sys
#import IMP
#import IMP.profile

import elements_constants
#import IMP.saxs
#import IMP.atom
import util
import particle


#import IMP.saxs
#import IMP.atom
#import IMP.util



"""

class FormFactorTable:

   class that deals with form factor computation
   two form factors are supported:
   (i) zero form factors for faster approximated calculations
   (ii) full form factors for slower accurate calculations

   Each form factor can be divided into two parts: vacuum and dummy.
   dummy is an approximated excluded volume (solvent) form factor.
   The approximation is done using Fraser, MacRae and Suzuki (1978) model.
   
"""

class FormFactorTable(object):

    def __init__(self,table_name=None):

        self.form_factor_type = {1:'ALL_ATOMS',2:'HEAVY_ATOMS',3:'CA_ATOMS',4:'RESIDUES'}
        #   H       He - periodic table line 1
        # Li  Be    B     C       N       O       F     Ne - line 2
        #  Na     Mg     Al     Si      P        S        Cl     Ar - line 3
        # K    Ca2+     Cr     Mn     Fe2+     Co - line 4
        # Ni   Cu      Zn2+     Se     Br - line 4 cont.
        # Ag    I       Ir     Pt      Au     Hg - some xelements from lines 5, 6
        # CH      CH2     CH3     NH       NH2       NH3     OH      OH2      SH
        self.vacuum_zero_form_factors = np.array([0.999953, 0.999872,2.99,
        3.99, 4.99, 5.9992, 6.9946, 7.9994, 8.99, 9.999,
        10.9924, 11.9865, 12.99, 13.99, 14.9993, 15.9998, 16.99, 17.99,
        18.99, 18.0025, 23.99, 24.99, 24.0006, 26.99,
        27.99, 28.99, 27.9996, 33.99, 34.99,46.99, 52.99, 76.99, 77.99, 78.9572, 79.99,
        6.99915, 7.99911, 8.99906, 7.99455, 8.99451, 9.99446, 8.99935, 9.9993,
        16.9998])
        
        self.element_dict = {'Unk':0,'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7, 'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15, 'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Cr':21,'Mn':22, 'Fe':23,
        'Co':24,'Ni':25,'Zn':26,'Se':27,'Br':28,'Ag':29,'I':30,'Ir':31,'Pt':32,'Au':33,'Hg':34,'CH':35,'CH2':36,
        'CH3':37,'NH':38,'NH2':39,'NH3':40,'OH':41,'OH2':42,'SH':43}
                             
                             
                         
        self.zero_form_factors = np.array([
        -0.720147, -0.720228,
        #   H       He - periodic table line 1
        1.591,     2.591,     3.591,   0.50824,  6.16294, 4.94998, 7.591,   6.993,
        # Li     Be      B     C       N        O       F      Ne - line 2
        7.9864,    8.9805,    9.984,   10.984,   13.0855, 9.36656, 13.984,  16.591,
        #  Na      Mg        Al       Si        P        S       Cl    Ar - line 3
        15.984,    14.9965,   20.984,  21.984,   20.9946, 23.984,
        # K       Ca2+       Cr      Mn      Fe2+      Co - line 4
        24.984,   25.984,     24.9936, 30.9825,  31.984,  43.984, 49.16,
        #Ni     Cu          Zn2+      Se       Br       Ag      I
        70.35676,  71.35676,  72.324,  73.35676,
        #Ir         Pt      Au      Hg
        -0.211907, -0.932054, -1.6522, 5.44279,  4.72265, 4.0025,  4.22983, 3.50968,
        8.64641])
        #  CH        CH2        CH3     NH       NH2       NH3     OH       OH2
        # SH


        self.dummy_zero_form_factors = np.array(

        [1.7201,  1.7201,  1.399,   1.399,   1.399,   5.49096, 0.83166, 3.04942, 1.399,   3.006,
        #  H     He     Li    Be    B       C        N        O      F?     Ne
        3.006,   3.006,   3.006,   3.006,   1.91382, 6.63324, 3.006,   1.399,
        # Na     Mg    Al    Si      P        S      Cl    Ar
        3.006,   3.006,   3.006,   3.006,   3.006,   3.006,
        # K   Ca2+    Cr    Mn   Fe2+   Co
        3.006,   3.006,   3.006,   3.006,   3.006,
        # Ni   Cu   Zn2+    Se     Br
        3.006,   3.83,    6.63324, 6.63324, 6.63324, 6.63324,
        # Ag   I       Ir      Pt       Au      Hg
        7.21106, 8.93116, 10.6513, 2.55176, 4.27186, 5.99196, 4.76952, 6.48962,
        8.35334])
        # CH      CH2     CH3     NH       NH2       NH3     OH      OH2      SH
        
        self.vanderwaals_radius = np.array(

        [1.20,1.40,1.82,1.93,1.92,1.70,1.55,1.52,1.47,1.54,
        # H He Li Be B C N O F Ne

        2.27,1.73,1.84, 2.10, 1.80, 1.80,1.75,1.88,
        # Na Mg Al Si P S Cl Ar

        2.75,2.31,2.00,2.00,2.00,2.00,
        # K Ca2+ Cr Mn Fe2+ Co

        1.63,1.40,1.39, 1.90,1.90,1.85,
        # Ni Cu Zn2+ Se Br

        1.72,1.98,2.00,1.75,1.66,1.55])
        # Ag I Ir Pt Au Hg
        
        self.element_dict = {'Unk':0,'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7, 'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15, 'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Cr':21,'Mn':22, 'Fe':23,
        'Co':24,'Ni':25,'Zn':26,'Se':27,'Br':28,'Ag':29,'I':30,'Ir':31,'Pt':32,'Au':33,'Hg':34,'CH':35,'CH2':36,
        'CH3':37,'NH':38,'NH2':39,'NH3':40,'OH':41,'OH2':42,'SH':43}

        self.vanderwaals_volume = None

        self.vacuum_form_factor_table = None
        self.dummy_form_factor_table = None
        self.form_factor_table = None

        self.rho = 0.334

        self.q = None
        self.min_q = 0.0
        self.max_q = 1.0
        self.delta_q = 0.1

        self.element_ff_dict = None
        self.residue_type_form_factor_dict_ = None

        self.form_factor_coefficients = None
        self.ff_table = None

        self.corrected_form_factors = None
        self.vacuum_form_factors = None
        self.dummy_form_factors = None

        # form factor coefficients (a's, b's,c's in particle.py)
        self.a_ = np.zeros((5, 1), dtype=np.float32)
        self.c_ = 0.0
        self.b = np.zeros((5, 1), dtype=np.float32)
        self.excl_vol = None #self.get_vanderwaals_volume()# calculated from volume of each element


        #self.init_element_form_factor_dict()
        #self.init_residue_type_form_factor_dict()
        #self.get_form_factor_coefficients()


        if table_name is not None:

            ff_num = self.read_form_factor_table(table_name)

            if ff_num > 0:

                for i in range(elements_constants.HEAVY_ATOM_SIZE):

                    self.zero_form_factors[i] = 0.0
                    self.dummy_zero_form_factors[i] = 0.0
                    self.vacuum_zero_form_factors[i] = 0.0

                number_of_q_entries = IMP.util.get_rounded(((self.q_max - self.q_min) / self.a_delta) + 1)
                form_factor_template = np.zeros(number_of_q_entries, dtype=np.float32)

                self.form_factors = [form_factor_template] * element_constants.HEAVY_ATOMS_SIZE

                self.vacuum_form_factors = [form_factor_template] * elements_constants.HEAVY_ATOM_SIZE

                self.dummy_form_factors = [form_factor_template] * elements_constants.HEAVY_ATOM_SIZE

        #self.compute_form_factors_all_atoms()
        #self.compute_form_factors_heavy_atoms()



    def read_form_factor_table(self,table_name):

        with open(table_name,'r') as fp:
            
            
            form_factors_coefficients_ = [AtomFactorCoefficients() for _ in range(elements_constants.ALL_ATOM_SIZE)]


            line = fp.readline()
            while line:

                if line[0] == '#':  # if comment line, read the whole line and move on

                    continue

                else:         #  return the first character
                    line = fp.readline()
                    break
                # read the data files
                #AtomFactorCoefficients coeff;

                counter = 0
                while not fp.eof():
                    line = fp.readline()
                    coeff = line.split(' ')
                    for j in coeff:

                        ff_type = atom_factor_coefficients.get_form_factor_atom_type(e)

                        if ff_type != 'UNK':
                            form_factors_coefficients_[ff_type] = coeff
                            counter += 1
                            print("read_form_factor_table: Atom type found: ",coeff.atom_type,'\n')
                        else:
                            print("Atom type is not supported ",coeff.atom_type,'\n')


                print(counter," form factors were read from file.",'\n')
        return counter

    def write_form_factor_table(self):
        print("hello from write form factor function")
        # TO DO

    def get_form_factor_atom_type(self, p,ff_type):

        print("hello from form factor atom type function ")
        # TO DO

    def get_form_factor(self, p, ff_type):
        
        ff_atom_type = self.get_form_factor_atom_type(p, ff_type)
        if ff_atom_type > elements_constants.HEAVY_ATOMS_SIZE:
            print("Atom type not found")

    def get_vacuum_zero_form_factors(self,i):

        return self.vacuum_zero_form_factors[i]


    def get_dummy_zero_form_factors(self,i):

        return self.dummy_zero_form_factors[i]

    def get_zero_form_factors(self,i):
        
        return self.zero_form_factors[i]


    def get_zero_form_factors(self):

        return self.zero_form_factors[i]

    def get_volume(self,p, ff_type='HEAVY_ATOMS'):

        form_factor = self.get_dummy_form_factors(p,ff_type)

        return form_factor/self.rho


    def get_vanderwaals_radius(self,i):

        return self.vanderwaals_radius[i]


    def get_vanderwaals_volume(self,i):
        
        return (4.0/3.0) * np.pi * np.power(self.vanderwaals_radius[i],3.0)

    def get_radius(self,i ,ff_type='HEAVY_ATOMS'):

        one_third = (1.0/3.0)
        c = (3.0/(4.0*np.pi))

        ii = i.astype(np.int32)
        print(ii)
        print (ii.shape)
        form_factor = self.get_dummy_zero_form_factors(ii-1)
        print(form_factor.shape)
        return np.power(c*form_factor, one_third)

    def get_dummy_form_factors(self):

        return self.dummy_form_factors

    def set_dummy_form_factors(self, dum):

        self.dummy_form_factors = dum

    def get_default_form_factor_table(self):

        return self.form_factor_table

    def init_element_form_factor_dict(self):

        self.element_ff_dict = {0:'H',1:'He',2:'Li', 3:'Be', 4:'B', 5:'C',6:'N',7:'O',8:'Fl',9:'Ne', 10:'Na',11:'Mg', 12:'Al',13:'Si',14:'P',
        15:'S', 16:'Cl',17:'Ar', 18:'K', 19:'Ca', 20:'Cr', 21:'Mn', 22:'Fe', 23:'Co',24:'Ni', 25:'Cu',26:'Zn',27:'Se',28:'Ag',29:'I',30:'Ir',31:'Au',32:'Hg'}

        return self.element_ff_dict

    def init_residue_type_form_factor_dict(self):

        self.residue_type_form_factor_dict_ = {(9.037, 37.991, 28.954):'ALA',
        (23.289, 84.972, 61.683):'ARG',(20.165, 58.989, 38.824):'ASP',
        (19.938, 59.985, 40.047):'ASN',(18.403, 53.991, 35.588):'CYS',
        (19.006, 67.984, 48.978):'GLN',(19.233, 66.989, 47.755):'GLU',
        (10.689, 28.992, 18.303):'GLY',(21.235, 78.977, 57.742):'HIS',
        (6.241, 61.989, 55.748):'ILE', (6.241, 61.989, 55.748):'LEU',
        (10.963, 70.983, 60.020):'LYS', (16.539, 69.989, 53.450):'MET',
        (9.206, 77.986, 68.7806):'PHE',(8.613, 51.9897, 43.377):'PRO',
        (13.987, 45.991, 32.004):'SER', (13.055, 53.99, 40.935):'THR',
        (14.156, 85.986, 71.83):'TYR', (14.945, 98.979, 84.034):'TRP',
        (7.173, 53.9896, 46.817):'VAL',(9.037, 37.991, 28.954):'UNK'}

    def get_form_factor_coefficients(self):
        
        particles = particle.Particle()
        #form_factor_coefficients
        self.form_factor_table = particles.get_ff_table()

        self.atom_type = particle.get_atom_type()

        a,b,c = particle.get_form_factor_coefficients()
   
        return a,c,b

        #return a,c,b
    def get_form_factor_table(self):

        return self.form_factor_table

    def calculate_corrected_form_factors(self,excl_vol,q_entries):

        self.corrected_form_factors = self.vacuum_form_factors - self.calculate_dummy_form_factors(excl_vol,q_entries)

        return self.corrected_form_factors

    def get_vacuum_form_factors(self):

        return self.vacuum_form_factors

    def set_vacuum_form_factors(self,fft):

        self.vacuum_form_factors = fft

    def get_dummy_form_factors(self):

        return self.dummy_form_factors

    def set_dummy_form_factors(self,dum):

        self.dummy_form_factors = dum

    def get_zero_form_factors(self):

        return self.zero_form_factors

    def set_zero_form_factors(self,z):

        self.zero_form_factors = z
        
    def get_water_form_factor():
        
        return self.dummy_form_factor[-2]


    def show(self):

        for i in range(constants_elements.HEAVY_ATOMS_SIZE):


            print('FFTYPE ',i,'zero_ff ',self.zero_form_factors[i], 'vacuum ff',self.vacuum_zero_ff[i], ' dummy ff', self.dummy_zero_form_factors[i],'\n')

    def calculate_dummy_form_factors(self,excluded,q_entries):

        #print excluded.dtype
        #print (q_entries)
        #print (q_entries.shape)



        #vol_coefficient = np.zeros((len(excluded),1),dtype=np.float64)
        #print vol_coefficient.shape
        #sys.exit()
        dff = np.zeros((len(excluded),len(q_entries)),dtype=np.float64)

        for i in range(len(excluded)):
            #vol_coefficient =
            #amp = self.rho * excluded[i]
            for iq in range(len(q_entries)):
                #vol_coefficient[i,iq ] = -np.power(excluded[i],(2.0,3.0))*(q_entries[iq]*q_entries[iq])/(16*np.pi)

                #print(vol_coefficient[i,iq])
                dff[i,iq] = self.rho * excluded[i] * np.exp(-np.power(excluded[i],(2.0/3.0)) * (q_entries[iq]**2)/(16.0*np.pi))
                
                #print dff.shape
        self.dummy_form_factors = dff
        #print(dff.shape)
        #print(dff)
        #print vol_coefficient
        #print vol_coefficient.shape
        #print len(q_entries)
        #for iq in range(len(q_entries)):
        return self.dummy_form_factors


    def compute_form_factors_all_atoms(self):
        #[a, cold,b] = self.get_form_factor_coefficients()
        self.q = np.linspace(0,10,101)
        number_of_q_entries = len(self.q)

        #number_of_q_entries = np.ceil((self.max_q - self.min_q)/self.delta_q).astype(np.int32)
        for i in range(1,elements_constants.ALL_ATOM_SIZE):

            #self.form_factor_coefficients[i].excl_vol = self.get_volume()

            self.vol_coefficient[i] = -np.power(self.get_vanderwaals_volume(i),(2.0,3.0))/(16*np.pi)


            for iq in range(number_of_q_entries):

                #self.q = self.min_q  + float(iq)*self.delta_q
                #s = self.q/(4.0*np.pi)
                
                

                #self.vacuum_form_factors_[i,iq] = 0.0 #self.form_factor_coefficients_[i,element_form_factor[i]]

                #for j in range(5):

                   #self.vacuum_form_factors_[i,iq] = self.a[j] * np.exp(-self.b_[j])



                self.dummy_form_factors_[i,iq] = self.rho_ * self.excl_vol_[i]* np.exp(self.vol_coeff * self.q_[iq] * self.q_[iq])

                #self.form_factors_[i,iq] = self.vacuum_form_factors_[i,iq] - self.dummy_form_factors_[i,iq]

            self.vacuum_form_factors = self.ff_table



            self.zero_form_factors[i] = c[i]

            #for j in range(5):

                #self.zero_form_factors_[i] += self.a_[j]

            self.vacuum_zero_form_factors[i] = self.zero_form_factors[i]

            self.dummy_zero_form_factors[i] = self.rho_ *  form_factor_coefficients_[i].excl_vol_

            self.zero_form_factors[i] -= self.rho * self.excl_vol_

# end of function: compute_form_factors_all_atoms

    def compute_form_factors_heavy_atoms(self):

        no_of_q_entries = np.ceil((self.max_q - self.min_q) / self.delta_q)

        IMP.FormFactorAtomType.set_element_type('Unk')
        h_num = 0

        for i in range(elements_constants.ALL_ATOM_SIZE,elements.constants.HEAVY_ATOM_SIZE):
            if i == self.element_dict['CH']:
            #if i == IMP.element_constants.CH:
                element_type = self.element_dict['C']
                #element_type = elements_constants.C
                h_num = 1
                
            elif i == self.element_dict['CH2']:
                element_type = self.element_dict['C']
            #elif i== IMP.elements_constants.CH2:
                #element_type = elements_constants.C
                h_num = 2

            #elif i== IMP.elements_constants.CH3:
            elif i==self.element_dict['CH3']:
                element_type = self.element_dict['C']
                #element_type = elements_constants.C
                h_num =  3

            #elif i== IMP.elements_constants.OH:
            elif i== self.element_dict['OH']:
                element_type = self.element_dict['O']
                #element_type = elements_constants.O
                h_num = 1
                
            elif i== self.element_dict['OH2']:                    
            #elif i== IMP.elements_constants.OH2:
                element_type = self.element_dict['O']
                #element_type = elements_constants.O
                h_num = 2
                
            elif i== self.element_dict['NH']:
            #elif i== IMP.elements_constants.NH:
                element_type = self.element_dict['N']
                #element_type = elements_constants.N
                h_num = 1
                
            elif i == self.element_dict['NH2']:
            #elif i== IMP.elements_constants.NH2:
                element_type = self.element_dict['N']
                # elements_constants.N
                h_num = 2
            
            elif i == self.element_dict['NH3']:
            #elif i== elements_constants.NH3:
                element_type = self.element_dict['N']
                #element_type = elements_constants.N
                h_num = 3
            
            elif i == self.element_dict['SH']:
            #elif i== elements_constants.SH:
                element_type = self.element_dict['S']
                #element_type = elements_constants.S
                h_num = 1

            # full form factor calculations
            
            for iq in range(self.number_of_q_entries):
            
                self.form_factors[iq] = self.form_factors[element_type,iq] + h_num*self.form_factors[self.element_dict['H'],iq]

                self.vacuum_form_factors[iq] = self.get_vacuum_form_factors[element_type,iq] + h_num *self.vacuum_form_factors[self.element_dict['H'],iq]

                self.dummy_form_factors[iq] = self.dummy_form_factors[element_type,iq] + h_num * self.dummy_form_factors[self.element_dict['H'],iq]

            # zero form factor calculations
            self.zero_form_factors_[i] = self.zero_form_factors[element_type] + h_num * self.zero_form_factors[self.element_dict['H']]
            self.vacuum_zero_form_factors[i] = self.vacuum_zero_form_factors[element_type] + h_num * self.vacuum_zero_form_factors[self.element_dict['H']]
            self.dummy_zero_form_factors[i] =  self.dummy_form_factors[element_type] + h_num * self.dummy_form_factors[self.element_dict['H']]

# end of function: compute_form_factor_heavy_atoms


"""
    compute form factors all atoms 

    f(q) = f_atomic(q) - f_solvent(q)
    f_atomic(q) = c + SUM[a_i * EXP(- b_i * (q / 4pi) ^ 2)]
    i = 1, 5


f_solvent(q) = rho * v_i * EXP((- v_i ^ (2 / 3) / (4pi)) * q ^ 2)

"""

"""
class AtomFactorCoefficients

a class for storing form factors solvation table


"""




