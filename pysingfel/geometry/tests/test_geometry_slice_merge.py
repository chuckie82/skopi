import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import time

import pysingfel as ps
import pysingfel.gpu as pg


def test_take_n_slice():
    dir_ = os.path.dirname(__file__) + '/..'

    # Load beam
    beam = ps.Beam(dir_+'/../../examples/input/exp_chuck.beam')

    # Load and initialize the detector
    det = ps.PnccdDetector(
        geom=dir_+'/../../examples/lcls/amo86615/'
             'PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
        beam=beam)

    mesh_length = 128
    mesh, voxel_length = det.get_reciprocal_mesh(
        voxel_number_1d=mesh_length)

    with h5.File('imStack-test.hdf5','r') as f:
        volume_in = f['volume'][:]
        slices_in = f['imUniform'][:]
        orientations_in = f['imOrientations'][:]

    slices_rec = ps.geometry.take_n_slice(
        pattern_shape = det.pedestal.shape,
        pixel_momentum = det.pixel_position_reciprocal,
        volume = volume_in,
        voxel_length = voxel_length,
        orientations = orientations_in)

    # Note: This does not work if orientations is stored as float32
    assert np.allclose(slices_in, slices_rec)
