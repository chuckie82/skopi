import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time
import pytest

import pysingfel as ps
import pysingfel.constants as cst
from pysingfel.geometry import slice_
from pysingfel.util import xp


class TestSlice(object):
    """Test slicing functions."""
    @classmethod
    def setup_class(cls):
        indices = xp.arange(8)
        eps = 1e-12
        cls.volume = (indices+1.).reshape((2, 2, 2))
        cls.voxel_length = 2. + eps

        # Create a momentum array that matches the voxels positions
        b0 = indices // 4
        b1 = (indices // 2) % 2
        b2 = indices % 2
        bin_indices = xp.vstack([b0, b1, b2])
        cls.pixel_momentum = (2*bin_indices-1).T.reshape(2, 2, 2, 3)

    def test_take_slice_center_value(self):
        """Test take_slice on center value."""
        orientation = cst.quat1
        pixel_momentum = xp.array([[[[0., 0., 0.]]]])
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               pixel_momentum, orientation)
        assert xp.isclose(slice_[0, 0, 0], self.volume.mean())

    def test_take_slice_default_orientation(self):
        """Test take_slice with the default orientation."""
        orientation = cst.quat1
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_, self.volume)

    def test_take_slice_x_orientation(self):
        """Test take_slice with 180 degrees x rotation."""
        orientation = cst.quatx
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_[:, ::-1, ::-1], self.volume)

    def test_take_slice_y_orientation(self):
        """Test take_slice with 180 degrees y rotation."""
        orientation = cst.quaty
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_[::-1, :, ::-1], self.volume)

    def test_take_slice_z_orientation(self):
        """Test take_slice with 180 degrees z rotation."""
        orientation = cst.quatz
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_[::-1, ::-1, :], self.volume)

    def test_take_slice_x90_orientation(self):
        """Test take_slice with 90 degrees x rotation."""
        orientation = cst.quatx90
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_[:, :, ::-1].swapaxes(1, 2), self.volume)

    def test_take_slice_x270_orientation(self):
        """Test take_slice with 270 degrees x rotation."""
        orientation = cst.quatx270
        slice_ = ps.take_slice(self.volume, self.voxel_length,
                               self.pixel_momentum, orientation)
        assert xp.allclose(slice_[:, ::-1, :].swapaxes(1, 2), self.volume)

    def test_take_slices(self):
        """Test take_slices with 90 & 270 degrees y & z rotations."""
        # Slices
        orientations = np.vstack(
            [cst.quaty90, cst.quaty270, cst.quatz90, cst.quatz270])
        slices = ps.take_n_slices(self.volume, self.voxel_length,
                                  self.pixel_momentum, orientations)
        assert xp.allclose(slices[0, ::-1, :, :].swapaxes(0, 2), self.volume)
        assert xp.allclose(slices[1, :, :, ::-1].swapaxes(0, 2), self.volume)
        assert xp.allclose(slices[2, :, ::-1, :].swapaxes(0, 1), self.volume)
        assert xp.allclose(slices[3, ::-1, :, :].swapaxes(0, 1), self.volume)
