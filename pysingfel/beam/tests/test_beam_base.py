import numpy as np
import pytest

import pysingfel as ps


WAVELENGTH = 1e-10
WAVENUMBER = 1e10
PHOTON_ENERGY = 1.23984197386209e4
DIM = 1e-7
FLUENCE = 1e12


def test_wavelength():
    beam = ps.Beam(wavelength=WAVELENGTH, focus_radius=DIM, fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH, atol=1e-14)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)


def test_wavenumber():
    beam = ps.Beam(wavenumber=WAVENUMBER, focus_radius=DIM, fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH, atol=1e-14)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)


def test_photon_energy():
    beam = ps.Beam(photon_energy=PHOTON_ENERGY, focus_radius=DIM,
                   fluence=FLUENCE)
    assert np.isclose(beam.wavelength, WAVELENGTH, atol=1e-14)
    assert np.isclose(beam.wavenumber, WAVENUMBER)
    assert np.isclose(beam.photon_energy, PHOTON_ENERGY)


def test_wavelength_wavenumber_clash():
    with pytest.raises(TypeError):
        beam = ps.Beam(wavelength=WAVELENGTH, wavenumber=WAVENUMBER,
                       focus_radius=DIM, fluence=FLUENCE)


def test_wavelength_photon_energy_clash():
    with pytest.raises(TypeError):
        beam = ps.Beam(wavelength=WAVELENGTH, photon_energy=PHOTON_ENERGY,
                       focus_radius=DIM, fluence=FLUENCE)


def test_wavenumber_photon_energy_clash():
    with pytest.raises(TypeError):
        beam = ps.Beam(wavenumber=WAVENUMBER, photon_energy=PHOTON_ENERGY,
                       focus_radius=DIM, fluence=FLUENCE)

def test_circle_radius():
    beam = ps.Beam(photon_energy=PHOTON_ENERGY, focus_radius=DIM,
                   fluence=FLUENCE)
    focus_x, focus_y, focus_shape = beam.get_focus()
    assert np.isclose(focus_x, 2*DIM, atol=1e-14)
    assert np.isclose(focus_y, 2*DIM, atol=1e-14)
    assert focus_shape == "circle"
    assert np.isclose(beam.get_focus_area(), np.pi * DIM**2, atol=1e-21)

def test_circle_diameter():
    beam = ps.Beam(photon_energy=PHOTON_ENERGY, focus_x=DIM,
                   focus_shape="circle", fluence=FLUENCE)
    focus_x, focus_y, focus_shape = beam.get_focus()
    assert np.isclose(focus_x, DIM, atol=1e-14)
    assert np.isclose(focus_y, DIM, atol=1e-14)
    assert focus_shape == "circle"
    assert np.isclose(beam.get_focus_area(), np.pi/4 * DIM**2, atol=1e-21)

def test_circle_double_diameter_clash():
    with pytest.raises(TypeError):
        beam = ps.Beam(photon_energy=PHOTON_ENERGY, focus_x=DIM,
                       focus_y=DIM, focus_shape="circle", fluence=FLUENCE)
