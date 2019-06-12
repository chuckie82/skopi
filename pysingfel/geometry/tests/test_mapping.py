import itertools
import numpy as np
import pytest

from pysingfel.geometry import mapping


def test_get_weight_and_index_center_odd():
    pixel_position = np.zeros((1,3), dtype=np.float)
    voxel_length = 1.
    voxel_num_1d = 5
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert np.all(index[0][0] == np.array([2, 2, 2]))
    assert weight[0][0] == 1
    assert np.all(weight[0][1:] == 0)


def test_get_weight_and_index_center_even():
    pixel_position = np.zeros((1,3), dtype=np.float)
    voxel_length = 1.
    voxel_num_1d = 4
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert np.all(index[0][0] == np.array([1, 1, 1]))
    assert np.allclose(weight[0], 0.125)
