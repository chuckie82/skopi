import numpy as np

import pysingfel as ps


WAVELENGTH = 1e-10
WAVENUMBER = 1e10
PHOTON_ENERGY = 1.23984197386209e4
DIM = 1e-7
FLUENCE = 1e12


def test_wavelength():
    beam = ps.Beam(wavelength=WAVELENGTH, focus_radius=DIM, fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)


def test_wavenumber():
    beam = ps.Beam(wavenumber=WAVENUMBER, focus_radius=DIM, fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)


def test_photon_energy():
    beam = ps.Beam(photon_energy=PHOTON_ENERGY, focus_radius=DIM,
                   fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)
