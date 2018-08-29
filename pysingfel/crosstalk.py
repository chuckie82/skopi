import numpy as np
from numba import jit


def make_cross_talk_lib(data_num=100000, effect_distance=1.,
                        boundary=2, average_adu=130, path='./cross_talk_lib.npy'):
    """
    Generate a numpy array to simulate the cross talk effect.
    
    Each small patch is a random sampling of a gaussian distribution.
    
    :param data_num: The number of small patches to generate
    :param effect_distance: The sigma of the gaussian distribution
    :param boundary: The effective range
    :param average_adu:
    :param path: The address to save the lib
    :return:
    """
    data_num = int(data_num)

    side = int(2 * boundary + 1)

    data = np.zeros((data_num, 2 + side * side))
    hit_point = np.random.rand(2, data_num) - 0.5

    xs, ys = np.meshgrid(np.array(range(-boundary, boundary + 1)),
                         np.array(range(-boundary, boundary + 1)))

    coordinate = np.zeros((side, side, 2))
    coordinate[:, :, 0] = xs
    coordinate[:, :, 1] = ys

    coordinate = np.reshape(coordinate, [side * side, 2])

    for l in range(data_num):
        distances = np.sum(np.square(coordinate - hit_point[np.newaxis, :, l]), axis=-1)
        data[l, 0:2] = hit_point[:, l]
        data[l, 2:] = distances

    # convert to density
    data[:, 2:] = np.exp(- data[:, 2:] / effect_distance)

    # normalize
    norm = np.sum(data[:, 2:], axis=-1)
    data[:, 2:] = data[:, 2:] / norm[:, np.newaxis]

    lib = np.zeros((data_num, side * side))
    for l in range(data_num):
        lib[l, :] = np.random.multinomial(average_adu, data[l, 2:])
    np.save(path, lib)

    print(" The cross talk effect library is saved to" + path)


@jit
def cross_talk_effect(dbase, photons, shape, dbsize, boundary):
    """
    Add crosstalk effect.

    :param dbase: The library of the cross talk effect.
    :param photons: The photons stack
    :param shape: The shape of the photon stack
    :param dbsize: The number of patterns in the database
    :param boundary: The effective region of the crosstalk effect
    :return: The modified pattern.
    """
    # Create the variable to hold the value
    adu = np.zeros((shape[0], shape[1] + boundary, shape[2] + boundary))

    for l in range(shape[0]):
        for m in range(shape[1]):
            for n in range(shape[2]):
                index = np.random.randint(low=0, high=dbsize, size=(photons[l, m, n],))
                adu[l, m:m + 5, n:n + 5] += np.sum(dbase[index, :], axis=0)
    return adu


def add_cross_talk_effect_panel(db_path, photons):
    """
    Add crosstalk effect
    :param db_path: The library of the cross talk effect.
    :param photons: The photons stack
    :return:
    """
    dbase = np.load(db_path)
    dbsize = dbase.shape[0]
    # calculate the boundary of the model
    boundary = np.sqrt(dbase.shape[1])
    boundary = boundary - 1
    boundary = int(boundary)

    library = np.reshape(dbase, [dbsize, boundary + 1, boundary + 1])
    # build the adu from the lib
    shape = photons.shape
    adu = cross_talk_effect(dbase=library, photons=photons, shape=shape, dbsize=dbsize, boundary=boundary)

    return adu[:, boundary / 2:-boundary / 2, boundary / 2:-boundary / 2]
