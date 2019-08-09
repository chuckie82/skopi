"""
 *  profile.py :  A class for profile storing and computation
 *
 *  Copyright 2007-2019 IMP Inventors. All rights reserved.
 *
 """
 
import numpy as np
import form_factor_table
#import pysingfel as ps
import sys,os
import io
import radial_distribution_function
import random
import time

class Profile:

    def __init__(self,qmin, qmax, delta,file_name=None,fit_file=False,max_q=3.0,units=1):

        self.modulation_function_parameter_ = 0.23
        self.SAXS_delta_limit = 1.0e-15
        self.iterations = 10
        self.min_q_ = qmin
        self.max_q_ = qmax
        self.delta_q_ = delta
        self.c1_ = 10
        self.c2_ = 10
        self.experimental_ = False
        self.average_radius_ = 1.58
        self.average_volume_  = 17.5
        self.id_ = 0
        self.beam_profile_ = None
        self.set_was_used = True
        #self.ff_table_ = form_factor_table.get_default_form_factor_table()
        self.sz = np.int(np.ceil((self.max_q_ - self.min_q_) / self.delta_q_) + 1)

        self.partial_profiles_ = None
        self.fit_file = fit_file
        self.intensity_ = None
        self.q_ = None
        self.error_ = None

        if self.fit_file and file_name is not None:

            self.experimental_  = False
            self.read_SAXS_file(file_name,fit_file,max_q,units)
            self.experimental_ = True
            self.name_ = file_name
            self.id_ = 0
            self.beam_profile_  = None
            self.set_was_used = True

    def get_q_size(self):
        
        return self.sz
    
    def get_max_q(self):
        
        return self.max_q_
    
    def set_max_q(self,mq):
        
        self.max_q_ = mq

    def get_min_q(self):
        
        return self.min_q_
    
    def set_min_q(self,mq):
        
        self.min_q_ = mq
        
    def get_delta_q(self):
        
        return self.delta_q_
    
    def set_delta_q(self,dq):
        
        self.delta_q_ = dq
        
    def get_experimental_flag(self):
    
        return self.experimental_
    
    def set_experimental_flag(self,ex):
        
        self.experimental_ = ex
      
    def get_average_radius():
        
        return self.average_radius_
    
    def set_average_radius(ar):
        
        self.average_radius = ar
        
    def get_average_volume():

        return self.average_volume_
    
    def set_average_volume(av):
        
        self.average_volume_ = av
        
    
    def initialize_param(self,size, partial_profiles_size):
        
        number_of_q_entries = size
        if number_of_q_entries == 0:
            number_of_q_entries = np.int(np.ceil((self.max_q_ - self.min_q_) / self.delta_q_) + 1)

        #self.q_ = np.zeros((number_of_q_entries,1),dtype=np.float32)
        #self.intensity_ = np.zeros((number_of_q_entries,1),dtype=np.float32)
        #self.error_ = np.zeros((number_of_q_entries,1),dtype=np.float32)
        #self.q_ = [None]*number_of_q_entries
        #self.intensity_ = [None]*number_of_q_entries
        #self.error_ = [None]*number_of_q_entries
        self.q_ = []
        self.intensity_ = []
        self.error_  = []
        self.partial_profiles = []
       
        if self.sz == 0:
            for i in range(number_of_q_entries):
                self.q_.append(self.min_q_ + i * self.delta_q_)
            self.q_ = np.array(self.q_)
            
        #if partial_profiles_size > 0:

             #self.partial_profiles = np.zeros((partial_profiles_size,number_of_q_entries),dtype=np.float32)


    def find_max_q(self,file_name):

        try:
            inputFile = open(file_name,'r')

        except IOError as ioe:
                print("Can't open file %s: %s \n" % (file_name,ioe))
                sys.exit()

        max_q = 0.0
        lines = inputFile.readlines()
        
        for line in lines:
       

            if line[0] == "#" or line[0] == "\0" or not line[0].isdigit():
                #print('Skip: %s\n' % line[0])
                continue

            split_results = None
            split_results = line.strip().split()
            #[s.strip() for s in line.splitlines()]
            sr = len(split_results)
            #print "sr=",sr,"\n"
            if sr < 2 or sr > 5:
                continue

            bad = set("1234567890.-+Ee")
            if set(split_results[0]) > bad:
                continue
                
            line = inputFile.readline()
        max_q = np.float(split_results[0])
        print max_q
        inputFile.close()
        return max_q

    # tested
    def read_SAXS_file(self,file_name, fit_file=None, max_q=0,units=1):


        try:
            inputFile = open(file_name,'r')
        except IOError as ioe:
            print("Can't open file %s,%s" % (file_name, ioe))


        # determine the data units as some files use 1/nm
        default_units = True # units=2 ==> 1/A
        print default_units
        if units == 3:
            default_units = False # units=3 ==> 1/nm
            # units=1 ==> unknown units, determine based on max_q
        if units == 1 and self.find_max_q(file_name) > 1.0:
            default_units = False

        with_error = False


        qs  = []
        intensities = []
        errors = []
        
        lines = inputFile.readlines()
        
        
        for line in lines:
            print line
            
            
            # skip comments
            #print line
            if line[0] == '#' or line[0] == '\0' or not line[0].isdigit():
                #print('Skip\n')
                continue
                
            split_results = line.strip().split()
            
            sr = len(split_results)
            #print(sr)
            
            
            if sr < 2 or sr > 5:
                continue

            first_no_num = set("1234567890.-+Ee")

            if set(split_results[0]) > first_no_num:
                continue #// not a number

            q = np.float(split_results[0])
            #print 'q=',q,'\n'
            
            if not default_units:
                q /= 10.0 # convert from 1/nm to 1/A

            if max_q > 0.0 and q > max_q:
                break  #stop reading after max_q

            if self.fit_file: # 4 columns: q, Iexp, err, Icomputed
                if sr is not 4:
                    continue
                intensity = np.float(split_results[3])
            else:
                intensity = np.float(split_results[1])


            # validity checks
            if np.abs(intensity) < self.SAXS_delta_limit:
                continue         # skip zero intensities
            if intensity < 0.0: #  negative intensity
                print("Warning: negative intensity value: %s,skipping remaining profile points\n" % line)
                break

            error = 1.0
            if sr >= 3 and not fit_file:
                error = np.float(split_results[2])
                if np.abs(error) < self.SAXS_delta_limit:
                    error = 0.05 * intensity
                    if np.abs(error) < self.SAXS_delta_limit: # Marc: this is strange, check C++ code
                         continue  # skip entry

                with_error = True

            qs.append(q)
            intensities.append(intensity)
            errors.append(error)

        inputFile.close()

        if len(qs) > 0:
            self.initialize_param(len(qs),6)
        
        self.q_ = np.array(qs)
        self.intensity_ = np.array(intensities)
        self.error_ = np.array(errors)
       
        # determine qmin, qmax and delta
        if self.sz > 1:
            self.min_q_ = self.q_[0]
            self.max_q_ = self.q_[self.sz - 1]
        else:
            print "Error: Number of q entries equals 1: exiting...\n" 
            sys.exit()

        if self.is_uniform_sampling():
        # To minimize rounding errors, by averaging differences of q
            diff = 0.0
            for i in range(1,self.sz):
                diff += self.q_[i] - self.q_[i - 1]
                self.delta_q_ = diff / (self.sz - 1)
        else:
            self.delta_q_ = (self.max_q_ - self.min_q_) / (sz - 1)

        print ("Read_SAXS_file: %s, size= %d, delta_q = %f, min_q = %f, max_q = %f\n" % (file_name,self.sz,self.delta_q_,self.min_q_,self.max_q_))

        # saxs_read: No experimental error specified, add errors
        if not with_error:
            self.add_errors()
            print("Read_SAXS_file: No experimental error specified -> error added.\n")
            
    # tested           
    def write_SAXS_file(self,file_name, max_q):

        try:
            outFile = open(file_name,'w')
        
        except IOError as ioe:
            print("Can't open file %s: %s\n" % (file_name, ioe))

        # header line
        outFile.write("# SAXS profile: number of points = %d, q_min = %f, q_max = %f " % (self.min_q_,self.max_q_,self.sz))

        #if max_q > 0:
        #    outFile.write(str(max_q))
        #else:
        outFile.write(str(self.max_q_))
        outFile.write(", delta_q =%f\n" % self.delta_q_)
        outFile.write("#    q    intensity ")

        if self.experimental_:
             outFile.write("   error")
        outFile.write("\n")

        #inputFile.setf(std::ios::fixed, std::ios::floatfield)
        # Main data
        for i in range(self.sz):
            if max_q > 0 and self.q_[i] > max_q:
                break
        
            s1 = self.q_[i]

            outFile.write("%10.8f " % s1)

            s2 = self.intensity_[i]
       
            outFile.write("%15.8f " % s2)

            if self.experimental_: # do not print error for theoretical profiles
               
                s3 = self.error_[i]
                outFile.write("%10.8f" % s3)

            outFile.write("\n")

        outFile.close()


    def add_errors(self):

        for i in range(self.sz):
            ra = np.abs(np.random.poisson(10) / 10.0 - 1.0) + 1.0

            # 3% of error, scaled by 5q + poisson distribution
            self.error_[i] = 0.03 * self.intensity_[i] * 5.0 * (self.q_[i] + 0.001) * ra

            self.experimental_ = True


    def add_noise(self,percentage):


        for i in range(self.sz):
            random_number = np.random.poisson(10) / 10.0 - 1.0
            #X% of intensity weighted by (1+q) + poisson distribution
            self.intensity_[i] += percentage * self.intensity_[i] * (self.q_[i] + 1.0) * random_number

    def is_uniform_sampling(self):

        if self.sz <= 1:
            return False

        curr_diff = self.q_[1] - self.q_[0]
        epsilon = np.float(curr_diff) / 10

        for i in range(2,self.sz):
            diff = self.q_[i] - self.q_[i - 1]
            if np.abs(curr_diff - diff) > epsilon:
                return False

        return True

    
    def calculate_I0(particles):
        
        I0 = 0.0
        for i in range(particles.size()):
            I0 += form_factor_table.get_vacuum_form_factor(particles,ff_type)
        return I0 * I0
      
    
    def read_partial_profiles(self,file_name):

        try:
            fp = open(file_name,'r')
        except IOError as ioe:
            print("Can't open file %s: %s\n" % (file_name, ioe))
        else:
            print("Opened file successfully")
            qs = []

            #std::vector<std::vector<double> > partial_profiles
            psize = 6
            # init
            #partial_profiles.insert(partial_profiles.begin(), psize, Vector<double>());
            partial_profiles = np.zeros((psize,1),dtype = np.float64)

            lines = fp.readlines()
            for line in lines:
                #line = line.strip(line) # remove all spaces
                #skip comments
                if line[0] == '#' or line[0] == '\0' or line[9].isdigit():
                    continue
                split_results = None
                split_results = line.strip().split()
                    #split_results = line.split(split_results, line, isspace() or istab())

                if split_results.size() is not 7:
                    continue
                    qs.append(np.float(split_results[0]))
                for i in range(6):
                    partial_profiles.append(np.float(split_results[i + 1]))


                fp.close()

                if len(qs) > 0:
                    self.initialize_param(len(qs), psize)

                for i in range(len(qs)):
                    self.q_[i] = qs[i]
                    self.intensity_[i] = 1 #will be updated by sum_partial_profiles
                    self.error_[i] = 1
                    #for j in range(6):
                    #    partial_profiles_[j,i] = partial_profiles[j,i]

                self.sum_partial_profiles(1.0, 0.0, False)

                #determine qmin, qmax and delta
                if self.sz > 1:
                    self.min_q_ = self.q_[0]
                    self.max_q_ = self.q_[self.sz - 1]

                if self.is_uniform_sampling():
                    # To minimize rounding errors, by averaging differences of q
                    diff = 0.0
                    for i in range(1,self.sz):
                        diff += self.q_[i] - self.q_[i - 1]
                        self.delta_q_ = diff / (self.sz - 1)
                else:
                    self.delta_q_ = (self.max_q_ - self.min_q_) / (self.sz - 1)
                    
                print("read_partial_profiles: filename= %s, size= %d, q_delta=%f,q_min=%f,"
                  "q_max=" % (file_name,self.sz,self.q_min_,self.q_max_))


    def write_partial_profiles(file_name):

        try:
            fp = open(file_name,'w')
        except IOError as ioe:
            print("Can't open file %s: %s" % (file_name, ioe))
        else:
            print("Opened file for writing successfully.")


        #header line
            fp.write("# SAXS profile: number of points = %d,  q_min = %f, q_max = %f, delta_q=%f\n" % (self.sz,self.min_q_,self.max_q_,self.delta_q_))


            fp.write("#    q    intensity \n")

            for i in range(self.sz):
                
                
                w1 = str(self.q_[i])
                fp.write("%10.5f " % w1)

                if self.partial_profiles.size() > 0:
                    for j in range(self.partial_profiles_.size()):
                        
                        fp.write("%15.8f ",self.partial_profiles_[j,i])

                else: # not a partial profile
                    fp.write("%f " % self.intensity_[i])
            if self.experimental_ : #  do not print error for theoretical profiles
                
                fp.write("%f " % self.error_[i])

        fp.write("\n")
        fp.close()


    def calculate_profile_partial(self,particles,surface,ff_type):

        print("start partial profile calculation for %f particles.\n " % particles.size())

        coordinates = particle.get_atom_pos()
        
        vacuum_ff = np.zeros((particles.size(),1),dtype=np.float32)

        dummy_ff  = np.zeros((particles.size(),1),dtype=np.float32)
        water_ff = 0.0

        for m in range(particles.size()):
            vacuum_ff[m] = form_factor_table.get_vacuum_form_factor(particles[m], ff_type)
            dummy_ff[m] = form_factor_table.get_dummy_form_factor(particles[m], ff_type)

            if surface.size() == particles.size():
                water_ff.resize(particles.size())
                wf = form_factor_table.get_water_form_factor()
                
        for n in range(particles.size()):
            water_ff[n] = surface[n] * wf


        r_size = 3

        if surface.size() == particles.size():
            r_size = 6
            
        r_dist = [RadialDistributionFunction() for _ in range(r_size)]
        dist = 0.0

        #iterate over pairs of atoms
        for i in range(coordinates.size()):
            for j in range(i+1,coordinates.size()):
                #dist = self.get_squared_distance
                
                dist = (coordinates[i,0]-coordinates[j,0])** 2 + (coordinates[i,1] - coordinates[j,1])**2
                + (coordinates[i,2]-coordinates[j,2])**2
                
                r_dist[0].add_to_distribution(dist,2 * self.vacuum_ff[i] * self.vacuum_ff[j])  # constant
                r_dist[1].add_to_distribution(dist,2 * self.dummy_ff[i] * self.dummy_ff[j]) # c1^2
                r_dist[2].add_to_distribution(dist,2 * (self.vacuum_ff[i] * self.dummy_ff[j] +
                                          self.vacuum_ff[j] * self.dummy_ff[i]))  # -c1
                
                if r_size > 3:
                    r_dist[3].add_to_distribution(dist,self.water_ff[i] * self.water_ff[j])  # c2^2
                    r_dist[4].add_to_distribution(dist,(vacuum_ff[i] * water_ff[j] +
                                            self.vacuum_ff[j] * self.water_ff[i])) # c2
                    r_dist[5].add_to_distribution(dist,2 * (self.water_ff[i] * self.dummy_ff[j] +
                       self.water_ff[j] * self.dummy_ff[i])) # -c1*c2

                # add autocorrelation part
                r_dist[0].add_to_distribution(dist,self.vacuum_ff[i]*self.vacuum_ff[i])
                r_dist[1].add_to_distribution(dist,self.dummy_ff[i]*self.dummy_ff[i])
                r_dist[2].add_to_distribution(dist,2 * self.vacuum_ff[i] * self.dummy_ff[i])

                if r_size > 3:
                    r_dist[3].add_to_distribution(dist,self.water_ff[i]*self.water_ff[i])
                    r_dist[4].add_to_distribution(dist,self.vacuum_ff[i] * self.water_ff[i])
                    r_dist[5].add_to_distribution(dist,2 * self.water_ff[i] * self.dummy_ff[i])

            # convert to reciprocal space
            self.squared_distributions_2_partial_profiles(r_dist)

            #compute default profile c1 = 1, c2 = 0
            self.sum_partial_profiles(1.0, 0.0, False)


    def sum_partial_profiles(self,c1=1.0,c2= 0.0, check = False):
        
        # precomputed exp function
        ef = np.exp(self.get_max_q(0)*self.get_max_q(0))

        if self.partial_profiles_.size() == 0:
            return

        # check if the profiles are already summed by this c1/c2 combination
        if check and np.abs(self.c1_ - c1) <= 0.000001 and np.abs(self.c2_ - c2) <= 0.000001:
            return

        # implements volume fitting function G(s) as described
        # in crysol paper eq. 13
        rm = self.average_radius_
        #this exponent should match the exponent of g(s) which doesn't have
        #(4pi/3)^3/2 part so it seems that this part is not needed here too.
        coefficient = np.power((4.0*np.pi/3.0), 2.0/3.0) * rm*rm * ((c1*c1-1.0)/(4*np.pi))

        square_c2 = c2 * c2
        cube_c1 = c1 * c1 * c1

        self.intensity_ = self.partial_profiles_[0]
        
        if self.partial_profiles.size() > 3:
            self.intensity_ += square_c2 * self.partial_profiles_[3]
            self.intensity_ += c2 * self.partial_profiles_[4]

        for k in range(self.sz):
            q = self.get_q(k)
            x = coefficient * q * q
            G_q = cube_c1
            if np.abs(x) > 1.0e-8:
                G_q *= ef.exp(x)
                G_q = cube_c1 * np.exp(coefficient*q*q)

                #self.[k] -= G_q * self.partial_profiles_[2,k]

            if self.partial_profiles_.size() > 3:
                self.intensity_[k] -= G_q * c2 * self.partial_profiles_[5,k]

        #cache new c1/c2 values
        self.c1_[count] = c1
        self.c2_[count] = c2
    
    # not tested
    def distribution_2_profile(self,r_dist):

        # iterate over intensity profile
        lq = linspace(self.q_min_,self.q_max_,(1.0/0.0001))
        
        for k in range(self.sz):
        #iterate over radial distribution
        
            for r in range(r_dist.size()):
                dist = r_dist.get_distance_from_index(r)
                x = dist * self.q_[k]*np.pi*lq
                xx = np.sinc(x)
                self.intensity_[k] += r_dist[r] * xx
    # not tested
    def squared_distribution_2_profile(self,r_dist):

        self.initialize_param(len(self.q_),len(r_dist))
        lq = linspace(self.q_min_,self.q_max_,(1.0/0.0001))
        sf = np.sinc(np.sqrt(r_dist.get_max_distance()) * self.get_max_q()*lq)

        distances = np.zeros((len(r_dist),1),dtype=np.float32)

        for r in range(len(r_dist)):
            if r_dist[r] is not 0.0:
                distances[r] = np.sqrt(r_dist.get_distance_from_index(r))

        use_beam_profile = False
        if self.beam_profile is not None and self.beam_profile_.size() > 0:
            self.use_beam_profile = True

            # iterate over intensity profile
            for k in range(self.sz):
                # iterate over radial distribution
                for r in range(r_dist.size()):
                    if r_dist[r] is not 0.0:
                        dist = distances[r]
                        x = 0.0
                #if self.use_beam_profile:
                    # iterate over beam profile
                #    for t in range(beam_profiles.size()):
                        # x = 2*I(t)*sinc(sqrt(q^2+t^2)) multiply by 2 because of the symmetry of the beam
                #       x1 = dist * np.sqrt((q_[k]*q_[k] + self.beam_profile.q_[t]*beam_profile.q_[t]))
                #       x += 2 * self.beam_profile.intensity_[t] * sf.sinc(x1)
                #else:
                    #x = np.sin(dq)/dq
                    lq = np.linspace(self.min_q_,self.max_q_,1.0/0.0001)
                    x = dist * self.q_[k]*lq
                    xx = sf.sinc(x)

                    # multiply by the value from distribution
                    self.intensity_[k] += r_dist[r] * xx


        # this correction is required since we approximate the form factor
        # as f(q) = f(0) * exp(-b*q^2)
        self.intensity_[k] *= np.exp(-self.modulation_function_parameter_ * self.q_[k]*self.q_[k])

    # not tested
    def squared_distributions_2_partial_profiles(self,r_dist,r_size):
        
        sq = len(self.q_)
        self.initialize_param(sq, r_size)
  
        lq = linspace(self.q_min,self.q_max_,1.0/(0.0001))
        n = np.sqrt(r_dist[0].get_max_distance())*get_max_q()*lq
        sf = np.sinc(n)

        # precompute square roots of distances
        distances = np.zeros((r_dist[0].size(),1),np.float32)

        for r in range(r_dist[0].size()):
            if r_dist[0,r] > 0.0:
                distances[r] = sqrt(r_dist[0].get_distance_from_index(r));


        #use_beam_profile = False
        #if beam_profile is not None and beam_profile.size() > 0:
        #    use_beam_profile = True

        # iterate over intensity profile
        for k in range(q_.size()):
        # iterate over radial distribution
            for r in range(len(r_dist[0])):
                if r_dist[0,r] > 0.0:
                    dist = distances[r]
                    x = 0.0

         #           if use_beam_profile:
                    # iterate over beam profile
         #               for t in range(beam_profile.size()):
                          # x = 2*I(t)*sinc(sqrt(q^2+t^2)) multiply by 2 because of the symmetry of the beam

         #                    x1 = dist * sqrt((q_[k]*q_[k] + beam_profile.q_[t]*beam_profile.q_[t]))
         #                    x += 2 * beam_profile_.intensity_[t] * sf.sinc(x1)

         #          else:
                        # x = sin(dq)/dq
                    x = dist * self.q_[k]
                    x = sf.sinc(x)
                     # iterate over partial profiles
                    for i in range(r_size):
                    # multiply by the value from distribution
                         self.partial_profiles_[i,k] += r_dist[i,r] * x

        # this correction is required since we approximate the form factor
        # as f(q) = f(0) * exp(-b*q^2)
            corr = np.exp(-modulation_function_parameter_ * self.q_[k]*self.q_[k])
        
        for i in range(r_size):
            self.partial_profiles_[i,k] *= corr

    # not tested
    def radius_of_gyration_fixed_q(self,end_q):
         # x=q^2, y=logI(q)) z=error(q)/I(q)
        errors = []
        data  = []
        for i in range(self.sz):
            q = self.q_[i]
            Iq = self.intensity_[i]
            err = self.error_[i] / Iq
            logIq = np.log(Iq)

            if q > end_q:
                break
            v.append((q * q, logIq))
            data.append(v)
            errors.append(err)
                     
        data = np.asarray(data)
        errors = np.asarray(errors)
                     
        #algebra::LinearFit2D lf(data, errors);
        lf = LinearFit2D(data,errors) 
        a = lf.get_a()
                     
        if a >= 0:
            return 0.0
                     
        rg = sqrt(-3 * a)
        return rg;


    # not tested
    def radius_of_gyration(self,end_q_rg):

        qlimit = self.min_q_ + self.delta_q_ * 5  # start after 5 points

        for q in range(qlimit,self.max_q_,self.delta_q_):

            rg = radius_of_gyration_fixed_q(q)
            if rg > 0.0:

                if q * rg < end_q_rg:
                    qlimit = q
            else:
                break


            rg = radius_of_gyration_fixed_q(qlimit)
        return rg

    # not tested
    def mean_intensity(self):

        mean = 0
        for i in range(self.sz):
            mean += self.intensity_[i]

        mean /= self.sz
        return mean
    # not tested
    def scale(self,c):
        for k in range(self.sz):
            self.intensity_[k] *= c
    # not tested
    def offset(self,c):
        for k in range(self.sz):
            self.intensity_[k] -= c

    def profile_2_distribution(self,rd,max_distance):

        scale = 1.0 / (2 * np.pi * np.pi)
        distribution_size = rd.get_index_from_distance(max_distance) + 1

        # offset profile so that minimal i(q) is zero
        min_value = self.intensity_[0]
        for k in range(self.sz):

            if self.intensity_[k] < min_value:
                min_value = self.intensity_[k]

            p = Profile(self.min_q_, self.max_q_, self.delta_q_)
            p.initialize_param(len(p),len(rd))
            
        for k in range(self.sz):
            p.intensity_[k] = self.intensity_[k] - min_value


            #iterate over r
        for i in range(len(rd)):
            r = rd.get_distance_from_index(i)
            sum = 0.0
            # sum over q: SUM (I(q)*q*sin(qr))
            for k in range(p.size()):
                sum += p.intensity_[k] * p.q_[k] * np.sin(p.q_[k] * r)

        rd.add_to_distribution(r, r * scale * sum)


                
        
                


