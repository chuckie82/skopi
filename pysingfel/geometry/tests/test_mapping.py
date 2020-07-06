import itertools
import numpy as np
import pytest

from pysingfel.geometry import mapping
from pysingfel.util import xp


def test_get_weight_and_index_center_odd():
    """Test get_weight_and_index centering for odd-sized meshes."""
    pixel_position = xp.zeros((1,3), dtype=np.float)
    voxel_length = 1.
    voxel_num_1d = 5
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert xp.all(index[0][0] == xp.array([2, 2, 2]))
    assert weight[0][0] == 1
    assert xp.all(weight[0][1:] == 0)


def test_get_weight_and_index_center_even():
    """Test get_weight_and_index centering for even-sized meshes."""
    pixel_position = xp.zeros((1,3), dtype=xp.float)
    voxel_length = 1.
    voxel_num_1d = 4
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert xp.all(index[0][0] == xp.array([1, 1, 1]))
    assert xp.allclose(weight[0], 0.125)


def test_get_weight_and_index_off_center():
    """Test get_weight_and_index for off-centered points."""
    pixel_position = xp.array([[0.1, 0.2, 0.3]])
    voxel_length = 2.
    voxel_num_1d = 5
    index, weight = mapping.get_weight_and_index(
        pixel_position, voxel_length, voxel_num_1d)
    assert xp.all(index[0][0] == xp.array([2, 2, 2]))
    assert xp.isclose(weight.sum(), 1.)
    assert xp.allclose(xp.dot(weight[0], index[0]),
                       xp.array([ 2.05, 2.1 , 2.15]))
