import numpy as np

class RadialDistributionFunction:

    def __init__(self,bin_sz=0.5):
        
   
        self.bin_size = bin_sz
    
        self.max_distance = 50.0

        self.vect3 = np.zeros((int(self.get_index_from_distance(self.max_distance)),1),dtype=np.float32)

        self.capacity = self.get_index_from_distance(self.max_distance + 1)
        
        self.r_size = 0
        
        self.index = 0
        
    
                     
    def get_index_from_distance(self,dist):
        #print dist
        #print self.bin_size
        dist += 0.001
        return np.rint((dist*1.0)/self.bin_size)
    
    def get_distance_from_index(self,index):
        print index
        print self.bin_size * index
        self.index = index
        return self.bin_size * index

    def get_max_distance(self):

        return self.max_distance

    def set_max_distance(self,md):

        self.max_distance = md

    def get_bin_size(self):

        return self.bin_size

    def set_bin_size(self,bin_sz):

        self.bin_size = bin_sz

    def add_to_distribution(self,dist,value):
        
        self.index = self.get_index_from_distance(dist,value)

        self.max_distance_ = self.get_distance_from_index(index+1)
        
        self.index += 1
        
        return self.max_distance,int(self.index)
        #self.index = index + 1
        #self.index += value
        #return 
        
        
            
        