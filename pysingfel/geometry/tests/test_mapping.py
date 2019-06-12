import itertools
import numpy as np
import pytest

from pysingfel.geometry import mapping


def test_get_weight_and_index_center_odd():
    """Test get_weight_and_index centering for odd-sized meshes."""
    pixel_position = np.zeros((1,3), dtype=np.float)
    voxel_length = 1.
    voxel_num_1d = 5
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert np.all(index[0][0] == np.array([2, 2, 2]))
    assert weight[0][0] == 1
    assert np.all(weight[0][1:] == 0)


def test_get_weight_and_index_center_even():
    """Test get_weight_and_index centering for even-sized meshes."""
    pixel_position = np.zeros((1,3), dtype=np.float)
    voxel_length = 1.
    voxel_num_1d = 4
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert np.all(index[0][0] == np.array([1, 1, 1]))
    assert np.allclose(weight[0], 0.125)


def test_get_weight_and_index_off_center():
    """Test get_weight_and_index for off-centered points."""
    pixel_position = np.array([[0.1, 0.2, 0.3]])
    voxel_length = 2.
    voxel_num_1d = 5
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert np.all(index[0][0] == np.array([2, 2, 2]))
    assert np.isclose(weight.sum(), 1.)
    assert np.allclose(np.dot(weight[0], index[0]),
                       np.array([ 2.05, 2.1 , 2.15]))
