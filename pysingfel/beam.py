import numpy as np


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


class Beam(object):
    """
    Basic beam object
    """

    def __init__(self, fname=None):
        """
        :param fname: The beam profile
        """
        self._wavelength = 0  # (m) wavelength
        self._photon_energy = 0  # (eV) photon energy
        self._wavenumber = 0  # (m^-1)
        self._focus_xFWHM = 0  # (m) beam focus diameter in x direction
        self._focus_yFWHM = 0  # (m) beam focus diameter in y direction
        self._focus_shape = 'circle'  # focus shape: {square, ellipse, default:circle}
        self._focus_area = 0  # (m^2)
        self._n_phot = 0  # number of photons per pulse
        self._phi_in = 0  # number of photon per pulse per area (m^-2)
        # Default polarization angle, requires input from user or file in the future
        self.Polarization = np.array([1, 0, 0])
        if fname is not None:
            self.read_beamfile(fname)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self._photon_energy = wavelength_to_photon_energy(self._wavelength)
        self._wavenumber = wavelength_to_wavenumber(self._wavelength)

    @property
    def photon_energy(self):
        return self._photon_energy

    @photon_energy.setter
    def photon_energy(self, value):
        self._photon_energy = value
        self._wavelength = photon_energy_to_wavelength(self._photon_energy)
        self._wavenumber = wavelength_to_wavenumber(self._wavelength)

    @property
    def wavenumber(self):
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, value):
        self._wavenumber = value
        self._wavelength = wavenumber_to_wavelength(self._wavenumber)
        self._photon_energy = wavelength_to_photon_energy(self._wavelength)

    def get_wavevector(self):
        """
        Get the wave vector. Notice that here, the wavevector is defined as
        [0, 0, 1 / wavelength]
        :return:
        """
        return np.array([0, 0, 2 * np.pi / self.wavelength])

    def set_focus(self, x, y=None, shape='circle'):
        """
        Set the focus of the beam.

        The shape variable defines the shape of the transverse profile of the beam.
        If don't know what to choose for the shape variable, leave it as default.
        The influence is very limited in this implementatin.

        :param x: The FWHM of the beam along x axis in meter.
        :param y: The FWHM of the beam along y axis in meter.
        :param shape:
        :return:
        """
        if y is not None:
            # ellipse
            self._focus_xFWHM = x
            self._focus_yFWHM = y
            shape = 'ellipse'
        else:
            # By default, circle
            self._focus_xFWHM = x
            self._focus_yFWHM = x
        self._focus_shape = shape
        self.set_focus_area()

    def get_focus(self):
        return self._focus_xFWHM

    def set_focus_area(self):
        if self._focus_shape is 'square':
            self._focus_area = self._focus_xFWHM * self._focus_yFWHM
        else:
            # Both ellipse and circle have the same equation.
            self._focus_area = np.pi/4 * self._focus_xFWHM * self._focus_yFWHM

    def get_focus_area(self):
        return self._focus_area

    def set_photons_per_pulse(self, x):
        """
        :param x: photons per pulse
        """
        self._n_phot = x

    def get_photons_per_pulse(self):
        return self._n_phot

    def get_photons_per_pulse_per_area(self):
        return self._n_phot / self._focus_area

    def read_beamfile(self, fname):
        """
        Read Beam file and set the corresponding property.

        :param fname: beam file.
        :return:
        """
        with open(fname) as f:
            content = f.readlines()
            for line in content:
                if line[0] != '#' and line[0] != ';' and len(line) > 1:
                    tmp = line.replace('=', ' ').split()
                    if tmp[0] == 'beam/photon_energy':
                        self.photon_energy = float(tmp[1])
                    if tmp[0] == 'beam/fluence':
                        self.n_phot = float(tmp[1])
                    if tmp[0] == 'beam/radius':
                        self.set_focus(float(tmp[1]))

    ####
    # Old-style setters and getters, for compatibility
    ####

    def set_wavelength(self, x):
        """
        :param x: wavelength in meter
        """
        self.wavelength = x

    def get_wavelength(self):
        return self.wavelength

    def set_photon_energy(self, ev):
        """
        Set photon energy
        :param ev: photon energy in ev
        """
        self.photon_energy = ev

    def get_photon_energy(self):
        return self.photon_energy

    def get_wavenumber(self):
        return self.wavenumber
