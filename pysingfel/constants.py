import numpy as np


Rx90 = np.array([
    [1., 0., 0.],
    [0., 0., -1.],
    [0., 1., 0.]])
Ry90 = np.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [-1., 0., 0.]])
Rz90 = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]])


quat1 = np.array([1., 0., 0., 0.])
quatx = np.array([0., 1., 0., 0.])
quaty = np.array([0., 0., 1., 0.])
quatz = np.array([0., 0., 0., 1.])
quatx90 = np.array([1., 1., 0., 0.]) / np.sqrt(2)
quaty90 = np.array([1., 0., 1., 0.]) / np.sqrt(2)
quatz90 = np.array([1., 0., 0., 1.]) / np.sqrt(2)
quatx270 = np.array([1., -1., 0., 0.]) / np.sqrt(2)
quaty270 = np.array([1., 0., -1., 0.]) / np.sqrt(2)
quatz270 = np.array([1., 0., 0., -1.]) / np.sqrt(2)


vecx = np.array([1., 0., 0.])
vecy = np.array([0., 1., 0.])
vecz = np.array([0., 0., 1.])
