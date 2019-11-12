
# form_factor_table.py

import numpy as np
import sys
import elements_constants
import util
import particle

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

    def __init__(self,table_name=None,mnq=0.0,mxq=3.0,dq=0.01):
    
        self.max_q_ = mxq
        self.min_q_ = mnq
        self.delta_q_ = dq
        

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
                             
                         #   H       He - periodic table line 1    
        
        self.ff_cm_map = { 'H':0,'HE':1,'C':2,'N':3,'O':4,'NE':5,'SOD+':6,'MG+2':7,'P':8,'S':9,'CAL2+':10,'FE2+':11,'ZN2+':12,'SE':13,'AU':14,'CH':15,'CH2':16,'CH3':17,'NH':18,'NH2':19,'NH3':20,'OH':21,'OH2':22,'SH':23}
        
        self.zero_form_factors = np.array([
        
        -0.720147, -0.720228,1.591,     2.591,     3.591,   0.50824,  6.16294, 4.94998, 7.591,   6.993,
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
        
        
        #UD                i=1,5
        #UD  a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5  excl_vol
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
        #print self.vol_coefficient.shape
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

        self.vacuum_form_factor_table = None
        self.dummy_form_factor_table = None
        self.form_factor_table = None

        self.rho = 0.334 # electron density of water

       
        self.element_ff_dict = None
        self.residue_type_form_factor_dict_ = None

        self.form_factor_coefficients = None
        self.ff_table = None

        self.corrected_form_factors = None
        self.vacuum_form_factors = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)
        self.dummy_form_factors = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)
        
        #self.excl_vol = None #self.get_vanderwaals_volume()# calculated from volume of each element

        self.init_element_form_factor_dict()
        #self.init_residue_type_form_factor_dict()
        #self.get_form_factor_coefficients()


        if table_name is not None:

            ff_num = self.read_form_factor_table(table_name)

            if ff_num > 0:

                for i in range(elements_constants.HEAVY_ATOM_SIZE):

                    self.zero_form_factors[i] = 0.0
                    self.dummy_zero_form_factors[i] = 0.0
                    self.vacuum_zero_form_factors[i] = 0.0

                number_of_q_entries = IMP.util.get_rounded(((self.q_max - self.q_min) / self.q_delta) + 1)
                form_factor_template = np.zeros(number_of_q_entries, dtype=np.float32)

                self.form_factors = [form_factor_template] * element_constants.HEAVY_ATOM_SIZE

                self.vacuum_form_factors = [form_factor_template] * elements_constants.HEAVY_ATOM_SIZE

                self.dummy_form_factors = [form_factor_template] * elements_constants.HEAVY_ATOM_SIZE

        self.compute_form_factors_all_atoms()
        #dummy = self.calculate_dummy_form_factors(self.excl_vol,self.number_of_q_entries)
        self.compute_dummy_form_factors()
        #print dummy
        self.compute_form_factors_heavy_atoms()
       

    def read_form_factor_table(self,table_name):

        with open(table_name,'r') as fp:
            
            
            #form_factors_coefficients_ = [AtomFactorCoefficients() for _ in range(elements_constants.ALL_ATOM_SIZE)]


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
        return self.form_factor[ff_atom_type]
        

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

        ii = i
        print(ii)
        
        form_factor = self.get_dummy_zero_form_factors(ii-1)
        print(form_factor.shape)
        return np.power(c*form_factor, one_third)

    def get_dummy_form_factors(self):

        return self.dummy_form_factors

    def set_dummy_form_factors(self, dum):

        self.dummy_form_factors = dum
        
    def get_vacuum_form_factors(self,part,ff_type='HEAVY_ATOMS'):
        #element = self.get_form_factor_atom_type(part,ff_type)
        return self.vacuum_form_factors
    
    def set_vacuum_form_factors(self,v):
        
        self.vacuum_form_factors = v

    def get_default_form_factor_table(self):

        return self.form_factor_table

    def init_element_form_factor_dict(self):

        self.element_ff_dict = {0:'H',1:'He',2:'Li', 3:'Be', 4:'B', 5:'C',6:'N',7:'O',8:'Fl',9:'Ne', 10:'Na',11:'Mg', 12:'Al',13:'Si',14:'P',
        15:'S', 16:'Cl',17:'Ar', 18:'K', 19:'Ca', 20:'Cr', 21:'Mn', 22:'Fe', 23:'Co',24:'Ni', 25:'Cu',26:'Zn',27:'Se',28:'Ag',29:'I',30:'Ir',31:'Au',32:'Hg'}

    def get_ff_cm_map(self):
        
        return self.ff_cm_map
    
    def get_element_dict_id(self,s):
        return self.element_ff_dict[s]
        
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
        
    def get_water_form_factor(self):
        
        return self.zero_form_factors[-2]


    def show(self):

        for i in range(40):#constants_elements.HEAVY_ATOMS_SIZE):


            print('FFTYPE ',i,'zero_ff ',self.zero_form_factors[i], 'vacuum ff',self.vacuum_zero_form_factors[i], ' dummy ff', self.dummy_zero_form_factors[i],'\n')

            

    def compute_dummy_form_factors(self):

  
        dff = np.zeros((self.ff_coeff.shape[0],self.number_of_q_entries),dtype=np.float64)

        for i in range(self.ff_coeff.shape[0]):
           
            for iq in range(self.number_of_q_entries):

                dff[i,iq] = self.rho * self.excl_vol[i] * np.exp(-np.power(self.excl_vol[i],(2.0/3.0)) * (self.q_[iq]**2)/(16.0*np.pi))
                
                
        self.dummy_form_factors = dff
    
        

    def calculate_dummy_form_factors(self,excluded,q_entries):


        dff = np.zeros((excluded.shape[0],q_entries.shape[0]),dtype=np.float64)

        for i in range(excluded.shape[0]):
            
            for iq in range(q_entries.shape[0]):
 
                dff[i,iq] = self.rho * excluded[i] * np.exp(-np.power(excluded[i],(2.0/3.0)) * (q_entries[iq]**2)/(16.0*np.pi))
                
                
        self.dummy_form_factors = dff
     
        return self.dummy_form_factors


    def compute_form_factors_all_atoms(self):
        
       
        s = np.zeros((self.number_of_q_entries,1),dtype=np.float64)
        for i in range(0,self.ff_coeff.shape[0]):


            
            for iq in range(self.number_of_q_entries):

                self.q_[iq] = self.min_q_  + float(iq)*self.delta_q_
                s[iq] = self.q_[iq]/(4.0*np.pi)
                
                

                self.vacuum_form_factors[i,iq] = self.c[i] #self.form_factor_coefficients_[i,element_form_factor[i]]

                for j in range(5):

                    self.vacuum_form_factors[i,iq] += self.a[i,j] * np.exp(-self.b[i,j]*s[iq]*s[iq])



            self.zero_form_factors[i] = self.c[i]

            for j in range(5):

                self.zero_form_factors[i] += self.a[i,j]


            self.zero_form_factors[i] -= self.rho * self.excl_vol[i]
        #print self.vacuum_form_factors
        #print "intermission"
        
            

# end of function: compute_form_factors_all_atoms
       
    
    def compute_form_factors_heavy_atoms(self):
        
        
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
        
        vacuum_form_factors_sh =  self.vacuum_form_factors[10] + 1.0*self.vacuum_form_factors[0]
        dummy_form_factors_sh =  self.dummy_form_factors[10] + 1.0 * self.dummy_form_factors[0]
        
        heavy_vacuum_form_factors = [vacuum_form_factors_ch,vacuum_form_factors_ch2,vacuum_form_factors_ch3,
            vacuum_form_factors_nh,vacuum_form_factors_nh2,vacuum_form_factors_nh3,
            vacuum_form_factors_oh,vacuum_form_factors_oh2,vacuum_form_factors_sh]
        heavy_dummy_form_factors = [dummy_form_factors_ch,dummy_form_factors_ch2,dummy_form_factors_ch3,
            dummy_form_factors_nh,dummy_form_factors_nh2,dummy_form_factors_nh3,
            dummy_form_factors_oh,dummy_form_factors_oh2,dummy_form_factors_sh]
        
        self.vacuum_form_factors = np.vstack((self.vacuum_form_factors,heavy_vacuum_form_factors))
        self.dummy_form_factors = np.vstack((self.dummy_form_factors,heavy_dummy_form_factors))
        #print self.vacuum_form_factors.shape
    
        
        
    '''
    def compute_form_factors_heavy_atoms(self):

        no_of_q_entries = np.ceil((self.max_q - self.min_q) / self.delta_q)

        
        h_num = 0

        for i in range(elements_constants.ALL_ATOM_SIZE,elements.constants.HEAVY_ATOM_SIZE):
            if i == self.element_dict['CH']:
                element_type = self.element_dict['C']
                h_num = 1
                
            elif i == self.element_dict['CH2']:
                element_type = self.element_dict['C']
                h_num = 2

            elif i==self.element_dict['CH3']:
                element_type = self.element_dict['C']
                h_num =  3

            elif i== self.element_dict['OH']:
                element_type = self.element_dict['O']
                h_num = 1
                
            elif i== self.element_dict['OH2']:                    
            
                element_type = self.element_dict['O']
                h_num = 2
                
            elif i== self.element_dict['NH']:
            
                element_type = self.element_dict['N']
                h_num = 1
                
            elif i == self.element_dict['NH2']:
            
                element_type = self.element_dict['N']
                h_num = 2
            
            elif i == self.element_dict['NH3']:
           
                element_type = self.element_dict['N']
                h_num = 3
            
            elif i == self.element_dict['SH']:
           
                element_type = self.element_dict['S']
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

    
    '''
    def get_carbon_atom_type(self,atom_type,residue_type):
    
        
        # protein atoms
        # CH
        if atom_type == 'CH':
           return 'CH'
        # CH2
        if atom_type == 'CH2':
           return 'CH2'
        # CH3
        if atom_type == 'CH3':
           return 'CH3'
        # C
        if atom_type == 'C':
           return 'C'

        # CA
        if atom_type == 'CA':
           if residue_type == 'GLY':
              return 'CH2'  # Glycine has 2 hydrogens
           return 'CH'
          
        # CB
        if atom_type == 'CB':
           if residue_type == 'ILE' or  residue_type == 'THR' or residue_type == 'VAL':
              print 'CH'
              
              return 'CH'
            
           if residue_type == 'ALA':
              return 'CH3'
           print 'CH2'
           
           return 'CH2'
          
        # CG
        if atom_type == 'CG':
           if residue_type == 'ASN' or residue_type == 'ASP' or residue_type == 'HIS' or residue_type == 'PHE' or residue_type == 'TRP' or residue_type == 'TYR':
              return 'C'
            
           if residue_type == 'LEU':
              return 'CH'
           return 'CH2'
          
        # CG1
        if atom_type == 'CG1':
           if residue_type == 'ILE':
              return 'CH2'
           if residue_type == 'VAL':
              return 'CH3'
          
        # CG2 - only VAL, ILE, and THR
        if atom_type == 'CG2':
           return 'CH3'
        
        # CD
        if atom_type == 'CD':
           if residue_type == 'GLU' or residue_type == 'GLN':
              return 'C'
           return 'CH2'
        
        # CD1
        if atom_type == 'CD1':
            if residue_type == 'LEU' or residue_type == 'ILE':
               return 'CH3'
            if residue_type == 'PHE' or residue_type == 'TRP' or residue_type == 'TYR':
               return 'CH'
            
            return 'C'
          
        # CD2
        if atom_type == 'CD2':
           if residue_type == 'LEU':
              return 'CH3'
           if residue_type == 'PHE' or residue_type == 'HIS' or residue_type =='TYR':
              return 'CH'
            
           return 'C'
          
        # CE
        if atom_type == 'CE':
        
           if residue_type == 'LYS':
              return 'CH2'
           if residue_type == 'MET':
           
              return 'CH3'
           return 'C'
          
        # CE1
        if atom_type == 'CE1':
           if residue_type == 'PHE' or residue_type == 'HIS' or residue_type == 'TYR':
              return 'CH'
            
           return 'C'
          
        # CE2
        if atom_type == 'CE2':
           if residue_type == 'PHE' or residue_type == 'TYR':
              return 'CH'
           return 'C'
        
        # CZ
        if atom_type == 'CZ':
           if residue_type == 'PHE':
              return 'CH'
           return 'C'
          
        # CZ1
        if atom_type == 'CZ1':
           return 'C'
           
        # CZ2, CZ3, CE3
        if atom_type == 'CZ2' or atom_type == 'CZ3' or atom_type == 'CE3':
           if residue_type == 'TRP':
              return 'CH'
           return 'C'
        # DNA/RNA atoms
        # C5'
        if atom_type == 'C5p':
           return 'CH2'
           #C1', C2', C3', C4'
           if atom_type == 'C4p' or atom_type == 'C3p' or atom_type == 'C2' or atom_type == 'C1p':
              return 'CH'
          
        # C2
        if atom_type == 'C2':
           if residue_type == 'DADE' or residue_type == 'ADE':
              return 'CH'
           
           return 'C'
          
        # C4
        if atom_type == 'C4':
           return 'C'
        
        # C5
        if atom_type == 'C5':
           if residue_type == 'DCYT' or residue_type == 'CYT' or residue_type == 'DURA' or residue_type == 'URA':
                return 'CH'
           return 'C'
          
        # C6
        if atom_type == 'C6':
           if residue_type == 'DCYT' or residue_type == 'CYT' or residue_type == 'DURA' or residue_type == 'URA' or residue_type == 'DTHY' or residue_type == 'THY':
              return 'CH'
           return 'C'
          
        # C7
        if atom_type == 'C7':
           return 'CH3'
        # C8
        if atom_type == 'C8':
           return 'CH'

        print "Carbon atom not found, using default C form factor for ",
        atom_type," ", residue_type,'\n'
        return 'C'
        

    def get_nitrogen_atom_type(self,atom_type,residue_type):
    
         # protein atoms
         #  N
         if atom_type == 'N':
            if residue_type == 'PRO':
                return 'N'
            return 'NH'
      
         
         # ND
         if atom_type == 'ND':
            return 'N'
            
         # ND1
         if atom_type == 'ND1':
            if residue_type == 'HIS':
     
                return 'NH'
            return 'N'
      
         # ND2
         if atom_type == 'ND2':
            if residue_type == 'ASN':
                return 'NH2'
            return 'N'
      
         # NH1, NH2
         if atom_type == 'NH1' or atom_type == 'NH2':
            if residue_type == 'ARG':
               return 'NH2'
            return 'N'
      
         # NE
         if atom_type == 'NE':
            if residue_type == 'ARG':
               return 'NH'
            return 'N'
      
         # NE1
         if atom_type == 'NE1':
            if residue_type == 'TRP':
                return 'NH'
            return 'N'
      
         # NE2
         if atom_type == 'NE2':
            if residue_type == 'GLN':
                return 'NH2'
            return 'N'
      
         # NZ
         if atom_type == 'NZ':
            if residue_type == 'LYS':
                return 'NH3'
            return 'N'
      

         # DNA/RNA atoms
         #N1
         if atom_type == 'N1':
            if residue_type == 'DGUA' or residue_type == 'GUA':
                return 'NH'
            return 'N'
      
         # N2, N4, N6
         if atom_type == 'N2' or atom_type == 'N4' or atom_type == 'N6':
            return 'NH2'
      
         # N3
         if atom_type == 'N3':
            if residue_type == 'DURA' or residue_type == 'URA':
                return 'NH'
            return 'N'
      
         # N7, N9
         if atom_type == 'N7' or atom_type == 'N9':
           return 'N'

      
         print "Nitrogen atom not found, using default N form factor for ",atom_type," ",residue_type,'\n'
                    
         return 'N'
        
    def get_sulfur_atom_type(self,atom_type,residue_type):
    
        # SD
        if atom_type == 'SD':
           return 'S'
        #SG
        if atom_type == 'SG':
           if residue_type == 'CYS':
              return 'SH'
           return 'S'
            
        print "Sulfur atom not found, using default S form factor for ", atom_type, " ",residue_type,'\n'
        return 'S'
        
        
    

    def get_oxygen_atom_type(self,atom_type,residue_type):
 
        # O OE1 OE2 OD1 OD2 O1A O2A OXT OT1 OT2
        if atom_type == 'O' or atom_type == 'OE1' or atom_type == 'OE2' or atom_type == 'OD1' or atom_type == 'OD2' or atom_type == 'O1A' or atom_type == 'O2A' or atom_type == 'OT1' or atom_type == 'OT2' or atom_type == 'OXT':
           return 'O'
   
        # OG
        if atom_type == 'OG':
            if residue_type == 'SER':
                return 'OH'
            return 'O'
   
        # OG1
        if atom_type == 'OG1':
            if residue_type == 'THR':
                return 'OH'
            return 'O'
   
        # OH
        if atom_type == 'OH':
            if residue_type == 'TYR':
                return 'OH'
            return 'O'
   
        # DNA/RNA atoms
        # O1P, O3', O2P, O2',O4',05', O2,O4,O6
        if atom_type == 'OP1' or atom_type == 'O3p' or atom_type == 'OP2' or atom_type == 'O2p' or atom_type == 'O4p' or  atom_type == 'O5p' or atom_type == 'O2' or atom_type == 'O4' or atom_type == 'O6':
           return 'O'
   
        #  O2'
        if atom_type == 'O2p':
           return 'OH'

        # water molecule
        if residue_type == 'HOH':
           return 'OH2'

        print "Oxygen atom not found, using default O form factor for ", atom_type, " ",residue_type,'\n'
                 
        return 'O'
 
    def get_form_factor_atom_type(self,atomic_type,atom_variant_type, residue_type):
       print atomic_type
       print atom_variant_type
       print residue_type
       
       
       if atomic_type == 'C':
       
          ret_type = self.get_carbon_atom_type(atom_variant_type, residue_type)
       elif atomic_type == 'N':
       
          ret_type = self.get_nitrogen_atom_type(atom_variant_type, residue_type)
       elif atomic_type == 'O':
       
          ret_type = self.get_oxygen_atom_type(atom_variant_type, residue_type)
       elif atomic_type == 'S':
       
          ret_type = self.get_sulfur_atom_type(atom_variant_type, residue_type)
       elif self.ff_cm_map.has_key(atomic_type):
          ret_type = atomic_type
       else:
          print "Can't find form factor for atom using default value of nitrogen \n"
          ret_type = 'N'
       print '\n\n'
       print ret_type
    
       return ret_type;
 

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

"""

                                  int number_of_q_entries = (int)std::ceil((max_q_ - min_q_) / delta_q_);

                                   // iterate over different atom types
                                   for (unsigned int i = 0; i < ALL_ATOM_SIZE; i++) {
                                     // form factors for all the q range
                                     // volr_coeff = - v_i^(2/3) / 4PI
                                     double volr_coeff =
                                         -std::pow(form_factors_coefficients_[i].excl_vol_, (2.0 / 3.0)) /
                                         (16 * PI);

                                     // iterate over q
                                     for (int iq = 0; iq < number_of_q_entries; iq++) {
                                       double q = min_q_ + (double)iq * delta_q_;
                                       double s = q / (4 * PI);

                                       // c
                                       vacuum_form_factors_[i][iq] = form_factors_coefficients_[i].c_;

                                       // SUM [a_i * EXP( - b_i * (q/4pi)^2 )] Waasmaier and Kirfel (1995)
                                       for (unsigned int j = 0; j < 5; j++) {
                                         vacuum_form_factors_[i][iq] +=
                                             form_factors_coefficients_[i].a_[j] *
                                             std::exp(-form_factors_coefficients_[i].b_[j] * s * s);
                                       }
                                       // subtract solvation: rho * v_i * EXP( (- v_i^(2/3) / (4pi)) * q^2  )
                                       dummy_form_factors_[i][iq] = rho_ *
                                                                    form_factors_coefficients_[i].excl_vol_ *
                                                                    std::exp(volr_coeff * q * q);

                                       form_factors_[i][iq] =
                                           vacuum_form_factors_[i][iq] - dummy_form_factors_[i][iq];
                                     }

                                     // zero form factors
                                     zero_form_factors_[i] = form_factors_coefficients_[i].c_;
                                     for (unsigned int j = 0; j < 5; j++) {
                                       zero_form_factors_[i] += form_factors_coefficients_[i].a_[j];
                                     }
                                     vacuum_zero_form_factors_[i] = zero_form_factors_[i];
                                     dummy_zero_form_factors_[i] =
                                         rho_ * form_factors_coefficients_[i].excl_vol_;
                                     // subtract solvation
                                     zero_form_factors_[i] -= rho_ * form_factors_coefficients_[i].excl_vol_;
                                   }
                                 }

"""




