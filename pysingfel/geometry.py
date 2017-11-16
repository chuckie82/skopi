import numpy as np
import math
import time

######################################################################
# Take slice from the volume
######################################################################
def take_one_slice(index_, weight_, volume_, pixel_num_):
    # Convert the index of the 3D diffraction volume to 1D
    volume_num_1d_ = volume_.shape[0]
    convertion_factor = np.array([volume_num_1d_*volume_num_1d_,volume_num_1d_,1],dtype = int)
    
    index_2d_ = np.reshape(index_, [pixel_num_, 8, 3])
    index_2d_ = np.matmul(index_2d_, convertion_factor)
    #index_2d_ = index_2d_.astype(np.int32)
    
    volume_1d_ = np.reshape(volume_, volume_num_1d_**3)
    weight_2d_ = np.reshape(weight_,[pixel_num_, 8])
    
    # Expand the data to merge
    data_to_merge_ = volume_1d_[index_2d_]
    
    # Merge the data
    data_merge_ = np.sum(np.multiply(weight_2d_,data_to_merge_),axis=1)
    
    return data_merge_.reshape(index_.shape[:3])

def take_n_slice(detector_, volume_, voxel_length_, orientations_ ):
    
    # Preprocess
    volume_num_1d = volume_.shape[0]
    pattern_shape_ = detector_.pixel_rms.shape
    number_ = orientations_.shape[0]
    pixel_position_ = detector_.pixel_position_reciprocal.copy()
    
    # Create variable to hold the slices
    slices_ = np.zeros((number_, pattern_shape_[0], pattern_shape_[1], pattern_shape_[2]))
    
    tic = time.time()
    for l in range(number_):
        # construct the rotation matrix
        rotmat_ = quaternion2rot3D(orientations_[l,:])
        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        pixel_position_ = rotate_pixels_in_reciprocal_space(rotmat_, pixel_position_)
        # calculate the index and weight in 3D
        index_ ,weight_ = get_weight_in_reciprocal_space(pixel_position_, voxel_length_, volume_num_1d)
        # get one slice
        slices_[l,:,:,:] = take_one_slice(index_, weight_, volume_ , detector_.pix_num_total)
        
    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds"%(number_, toc-tic))
    
    return slices_

def take_n_random_slices(detector_, volume_, voxel_length_, number_ ):
    
    # Preprocess
    volume_num_1d = volume_.shape[0]
    pattern_shape_ = detector_.pixel_rms.shape
    pixel_position_ = detector_.pixel_position_reciprocal.copy()
    
    # Create variable to hold the slices
    slices_ = np.zeros((number_, pattern_shape_[0], pattern_shape_[1], pattern_shape_[2]))
    
    tic = time.time()
    for l in range(number_):
        # construct the rotation matrix
        rotmat_ = quaternion2rot3D(getRandomRotation('x'))
        # rotate the pixels in the reciprocal space. Notice that at this time, the pixel position is in 3D
        pixel_position_new = rotate_pixels_in_reciprocal_space(rotmat_, pixel_position_)
        # calculate the index and weight in 3D
        index_ ,weight_ = get_weight_in_reciprocal_space(pixel_position_new, voxel_length_, volume_num_1d)
        # get one slice
        slices_[l,:,:,:] = take_one_slice(index_, weight_, volume_, detector_.pix_num_total)
        
    toc = time.time()
    print("Finishing constructing %d patterns in %f seconds"%(number_, toc-tic))
    
    return slices_


######################################################################
# The following functions are utilized to get corrections
######################################################################
def reshape_pixels_position_arrays_to_1d(array):
    
    array_1d = np.reshape(array, [np.prod(array.shape[:-1]),3])
    return array_1d

def _reciprocal_space_pixel_position(pixel_center, wave_vector, polarization):
    
    ## reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)
    
    ## Calculate the reciprocal position of each pixel
    wave_vector_norm = np.sqrt(np.sum(np.square(wave_vector)))
    wave_vector_direction = wave_vector/ wave_vector_norm
    
    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d),axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:,np.newaxis]
    
    pixel_position_reciprocal_1d = wave_vector_norm*(pixel_center_direction - wave_vector_direction)
    
    ## restore the pixels shape
    pixel_position_reciprocal = np.reshape(pixel_position_reciprocal_1d, pixel_center.shape)
    
    return pixel_position_reciprocal

def _polarization_correction(pixel_center, wave_vector, polarization):
    
    ## reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)
    
    #print pixel_center_1d.shape
    
    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d),axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:,np.newaxis]
    
    ## Calculate the polarization correction
    polarization_norm = np.sqrt(np.sum(np.square(polarization)))
    polarization_direction = polarization/polarization_norm

    
    polarization_correction_1d = np.sum(np.square(np.cross(pixel_center_direction, polarization_direction)),axis=1)
    
    #print polarization_correction_1d.shape
    
    polarization_correction = np.reshape(polarization_correction_1d, pixel_center.shape[0:-1])
    
    return polarization_correction
    
def _geometry_correction(pixel_center, orientation):
    
    ## reshape the array into a 1d position array
    pixel_center_1d = reshape_pixels_position_arrays_to_1d(pixel_center)
    distance = pixel_center_1d[0,2]
    pixel_center_norm = np.sqrt(np.sum(np.square(pixel_center_1d),axis=1))
    pixel_center_direction = pixel_center_1d / pixel_center_norm[:,np.newaxis]
    
    ## Calculate the solid angle correction
    orientation_norm = np.sqrt(np.sum(np.square(orientation)))
    orientation_normalized = orientation/orientation_norm
    
    ## The correction induced by the orientation
    geometry_correction_1d = np.abs(np.dot(pixel_center_direction, orientation_normalized))
    ## The correction induced by the distance
    distance_correction = np.square(distance/pixel_center_norm)
    geometry_correction_1d = np.multiply(geometry_correction_1d, distance_correction)
    
    geometry_correction = np.reshape(geometry_correction_1d, pixel_center.shape[0:-1])
    
    return geometry_correction
    
def reciprocal_space_pixel_position_and_correction(pixel_center, wave_vector, polarization, orientation):

    pixel_position_reciprocal = _reciprocal_space_pixel_position(pixel_center, wave_vector, polarization)
    pixel_position_reciprocal_norm = np.sqrt(np.sum(np.square(pixel_position_reciprocal),axis=-1))*( 1e-10/2. )
    
    polarization_correction = _polarization_correction(pixel_center, wave_vector, polarization)
    geometry_correction = _geometry_correction(pixel_center, orientation)
    
    return pixel_position_reciprocal, pixel_position_reciprocal_norm, polarization_correction, geometry_correction

######################################################################
# The following functions are utilized to get reciprocal space grid mesh
######################################################################

def get_reciprocal_mesh(voxel_num_1d, voxel_length):
    
    voxel_half_num_1d = (voxel_num_1d-1)/2
    
    x_meshgrid = (np.array(range(voxel_num_1d)) - voxel_half_num_1d)*voxel_length
    reciprocal_mesh_stack = np.meshgrid(x_meshgrid, x_meshgrid, x_meshgrid )  

    reciprocal_mesh= np.zeros((voxel_num_1d, voxel_num_1d, voxel_num_1d, 3))
    for l in range(3):
        reciprocal_mesh[:,:,:,l] = reciprocal_mesh_stack[l][:,:,:]
    
    return reciprocal_mesh 
    
def get_weight_in_reciprocal_space(pixel_position_reciprocal, voxel_length, voxel_num_1d):
    
    ##convert_to_voxel_unit 
    pixel_position_reciprocal_voxel = pixel_position_reciprocal / voxel_length
    voxel_half_num_1d = int(voxel_num_1d/2)
    
    ## Get the indexes of the eight nearest points.
    num_panel, num_x, num_y, _ = pixel_position_reciprocal.shape
    _indexes = np.zeros((num_panel, num_x, num_y, 2 ,3))
    for l in range(3):
        _indexes[:,:,:,0,l] = (np.floor(pixel_position_reciprocal_voxel[:,:,:,l])
                                                         + voxel_half_num_1d).astype('int')
        _indexes[:,:,:,1,l] = (np.floor(pixel_position_reciprocal_voxel[:,:,:,l])
                                                         + voxel_half_num_1d +1).astype('int')
                
    indexes = np.zeros((num_panel,num_x, num_y, 8, 3))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                indexes[:,:,:, l*4+m*2+n,0] = _indexes[:,:,:,l,0] 
                indexes[:,:,:, l*4+m*2+n,1] = _indexes[:,:,:,m,1] 
                indexes[:,:,:, l*4+m*2+n,2] = _indexes[:,:,:,n,2] 
    
    del _indexes
    
    difference = indexes - pixel_position_reciprocal_voxel[:,:,:, np.newaxis, :]
    distance = np.sqrt(np.sum(np.square(difference),axis=-1))
    
    del difference
    
    summation = np.sum(distance,axis=-1)
    weight = distance/summation[:,:,:,np.newaxis]
    
    return indexes.astype(np.int), weight

######################################################################
# The following functions are utilized to assemble the images
######################################################################

def assemble_image_from_index_and_panel(image_stack, index):
    # get boundary
    index_max_x = np.max(index[:,:,:,0])
    index_max_y = np.max(index[:,:,:,1])
    # set holder
    image = np.zeros((index_max_x, index_max_y))
    # loop through the panels
    for l in range(index.shape[0]):
        image[index[l,:,:,:]] = image_stack[l,:,:]
        
    return image

def batch_assemble_image_from_index_and_panel(image_stack, index):
    pass

######################################################################
# The following functions are utilized to rotate the pixels in reciprocal space
######################################################################

def rotate_pixels_in_reciprocal_space(rot_mat, pixels_position):
    pixels_position_1d = reshape_pixels_position_arrays_to_1d(pixels_position)
    pixels_position_1d = pixels_position_1d.dot(rot_mat)
    return np.reshape(pixels_position_1d, pixels_position.shape)


######################################################################
# Some trivial geometry calculations
######################################################################

def corrCoeff(X, Y):
    x = X.reshape(-1)
    y = Y.reshape(-1)
    x -= np.mean(x)
    y -= np.mean(y)
    return np.dot(x, y) / np.sqrt(np.dot(x, x)*np.dot(y, y))


# Converters between different descriptions of 3D rotation.
def angleAxis2rot3D(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.
    """
    if len(axis) is not 3:
        raise ValueError('Number of axis element must be 3!')
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    cosTheta = np.cos(theta)
    bracket = 1 - cosTheta
    aBracket = a * bracket
    bBracket = b * bracket
    cBracket = c * bracket
    sinTheta = np.sin(theta)
    aSinTheta = a * sinTheta
    bSinTheta = b * sinTheta
    cSinTheta = c * sinTheta
    rot3D = np.array([[a*aBracket+cosTheta, a*bBracket-cSinTheta, a*cBracket+bSinTheta],
                     [b*aBracket+cSinTheta, b*bBracket+cosTheta, b*cBracket-aSinTheta],
                     [c*aBracket-bSinTheta, c*bBracket+aSinTheta, c*cBracket+cosTheta]])
    return rot3D


def euler2rot3D(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to a rotation matrix in 3D.
    """
    Rphi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    Rtheta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]])
    Rpsi = np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])
    return np.dot(Rpsi, np.dot(Rtheta, Rphi))


def euler2quaternion(psi, theta, phi):
    """
    Convert rotation with euler angle (psi, theta, phi) to quaternion description.
    """
    if abs(psi) == 0 and abs(theta) == 0 and abs(phi) == 0:
        quaternion = np.array([1., 0., 0., 0.])
    else:
        R = euler2rot3D(psi, theta, phi)
        W = np.array([R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0]])
        if W[0] >= 0:
            W /= np.linalg.norm(W)
        else:
            W /= np.linalg.norm(W) * -1
        theta = np.arccos(0.5 * (np.trace(R) - 1))
        CCisTheta = corrCoeff(R, angleAxis2rot3D(W, theta))
        CCisNegTheta = corrCoeff(R, angleAxis2rot3D(W, -theta))
        if CCisNegTheta > CCisTheta:
            theta = -theta
        quaternion = np.array([np.cos(theta/2.), np.sin(theta/2.)*W[0], np.sin(theta/2.)*W[1], np.sin(theta/2.)*W[2]])
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def quaternion2AngleAxis(quaternion):
    """
    Convert quaternion to a right hand rotation theta about certain axis.
    """
    HA = np.arccos(quaternion[0])
    theta = 2 * HA
    if theta < np.finfo(float).eps:
        theta = 0
        axis = np.array([1, 0, 0])
    else:
        axis = quaternion[[1, 2, 3]] / np.sin(HA)
    return theta, axis


def quaternion2rot3D(quaternion):
    """
    Convert quaternion to a rotation matrix in 3D.
    Use zyz convention after Heymann (2005)
    """
    theta, axis = quaternion2AngleAxis(quaternion)
    return angleAxis2rot3D(axis, theta)


# Functions to generate rotations for different cases: uniform(1d), uniform(3d), random.
def pointsOn1Sphere(numPts, rotationAxis):
    """
    Given number of points and axis of rotation, distribute evenly on the surface of a 1-sphere (circle).
    """
    points = np.zeros((numPts, 4))
    incAng = 360. / numPts
    myAng = 0
    if rotationAxis == 'y':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, myAng * np.pi / 180, 0)
            myAng += incAng
    elif rotationAxis == 'z':
        for i in range(numPts):
            points[i, :] = euler2quaternion(0, 0, myAng * np.pi / 180)
            myAng += incAng
    return points


def pointsOn4Sphere(numPts):
    """
    Given number of points, distribute evenly on hyper surface of a 4-sphere.
    """
    points = np.zeros((2*numPts, 4))
    N = 4
    surfaceArea = N * np.pi ** (N/2) / (N/2)  # for even N
    delta = np.exp(np.log(surfaceArea / numPts) / 3)
    Iter = 0
    ind = 0
    maxIter = 1000
    while ind != numPts and Iter < maxIter:
        ind = 0
        deltaW1 = delta
        w1 = 0.5 * deltaW1
        while w1 < np.pi:
            q0 = np.cos(w1)
            deltaW2 = deltaW1 / np.sin(w1)
            w2 = 0.5 * deltaW2
            while w2 < np.pi:
                q1 = np.sin(w1) * np.cos(w2)
                deltaW3 = deltaW2 / np.sin(w2)
                w3 = 0.5 * deltaW3
                while w3 < 2 * np.pi:
                    q2 = np.sin(w1) * np.sin(w2) * np.cos(w3)
                    q3 = np.sin(w1) * np.sin(w2) * np.sin(w3)
                    points[ind, :] = np.array([q0, q1, q2, q3])
                    ind += 1
                    w3 += deltaW3
                w2 += deltaW2
            w1 += deltaW1
        delta *= np.exp(np.log(float(ind) / numPts) / 3)
        Iter += 1
    return points[0:numPts, :]


def getRandomRotation(rotationAxis):
    """
    Generate random rotation.
    """
    if rotationAxis == 'y':
        u = np.random.random() * 2 * np.pi  # random angle between [0, 2pi]
        return euler2quaternion(0, u, 0)
    else:
        u = np.random.rand(3)  # uniform random distribution in the [0,1] interval
        # generate uniform random quaternion on SO(3)
        return np.array([np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]), np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
                         np.sqrt(u[0]) * np.sin(2*np.pi*u[2]), np.sqrt(u[0]) * np.cos(2*np.pi*u[2])])

######################################################################
# Obsolete
######################################################################

'''
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
                     
                     
def convert_to_poisson(dp):
    """
    Add poisson noise to a certain diffraction pattern dp.
    """
    return np.random.poisson(dp)


'''



