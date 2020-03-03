def wavelength_to_wavenumber(wavelength):
    """
    Wavenumber is defined as 1/wavelength

    :param wavelength: wavelength in meter
    :return:
    """
    return 1. / wavelength


def wavenumber_to_wavelength(k):
    """

    :param k: The wavenumber in meter^-1.  k= 1./wavelength.
    :return: 1./k
    """
    return 1. / k


def photon_energy_to_wavelength(photon_energy):
    """
    Conver photon energy in ev to wave length in m.
    :param photon_energy: photon energy in ev
    :return:
    """
    return 1.23984197386209e-06 / photon_energy


def wavelength_to_photon_energy(wavelength):
    """
    Convert wave length to photon energy in ev
    :param wavelength: wavelength in m.
    :return:
    """
    return 1.23984197386209e-06 / wavelength
