import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time

import pysingfel as ps
from pysingfel.geometry import slice_


def test_take_n_slice():
    ex_dir_ = os.path.dirname(__file__) + '/../../../examples'

    # Load beam
    beam = ps.Beam(ex_dir_+'/input/exp_chuck.beam')

    # Load and initialize the detector
    det = ps.PnccdDetector(
        geom=ex_dir_+'/input/geometry/0-end.data',
        beam=beam)

    mesh_length = 128
    mesh, voxel_length = det.get_reciprocal_mesh(
        voxel_number_1d=mesh_length)

    with h5.File(ex_dir_+'/input/imStack-test.hdf5','r') as f:
        volume_in = f['volume'][:]
        slices_in = f['imUniform'][:]
        orientations_in = f['imOrientations'][:]

    slices_rec = slice_.take_n_slices(
        volume=volume_in,
        voxel_length=voxel_length,
        pixel_momentum=det.pixel_position_reciprocal,
        orientations=orientations_in)

    # Note: This does not work if orientations is stored as float32
    assert np.allclose(slices_in, slices_rec)
