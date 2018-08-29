import numpy as np


# several converters
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

    def __init__(self, *fname):
        """
        :param fname: The beam profile
        """
        self.wavelength = 0  # (m) wavelength
        self.photon_energy = 0  # (eV) photon energy
        self.k = 0  # (m^-1)
        self.focus_xFWHM = 0  # (m) beam focus diameter in x direction
        self.focus_yFWHM = 0  # (m) beam focus diameter in y direction
        self.focus_shape = 'circle'  # focus shape: {square, ellipse, default:circle}
        self.focus_area = 0  # (m^2)
        self.n_phot = 0  # number of photons per pulse
        self.phi_in = 0  # number of photon per pulse per area (m^-2)
        self.Polarization = 0  # Default polarization angle, requires input from user or file in the future
        if fname is not None:
            self.read_beamfile(fname[0])

    def update(self):
        """
        Update the related properties of the class object, after
        a certain property is provided.
        """
        if self.wavelength != 0:
            self.photon_energy = wavelength_to_photon_energy(self.wavelength)
            self.k = wavelength_to_wavenumber(self.wavelength)
        elif self.photon_energy != 0:
            self.wavelength = photon_energy_to_wavelength(self.photon_energy)
            self.k = wavelength_to_wavenumber(self.wavelength)
        elif self.k != 0:
            self.wavelength = wavenumber_to_wavelength(self.k)
            self.photon_energy = wavelength_to_photon_energy(self.wavelength)

        if self.focus_xFWHM != 0:
            self.set_focus_area()
            if self.n_phot != 0:
                self.set_photons_per_pulse_per_area()

    # setters and getters
    def set_wavelength(self, x):
        """
        :param x: wavelength in meter
        """
        self.wavelength = x
        self.update()

    def get_wavelength(self):
        return self.wavelength

    def set_photon_energy(self, ev):
        """
        Set photon energy
        :param ev: photon energy in ev
        """
        self.photon_energy = ev
        self.update()

    def get_photon_energy(self):
        return self.photon_energy

    def get_wavenumber(self):
        return self.k

    def get_wavevector(self):
        """
        Get the wave vector. Notice that here, the wavevector is defined as
        [0, 0, 2 * pi / wavelength]
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
            self.focus_xFWHM = x
            self.focus_yFWHM = y
            shape = 'ellipse'
        else:
            # By default, circle
            self.focus_xFWHM = x
            self.focus_yFWHM = x
        self.focus_shape = shape
        self.update()

    def get_focus(self):
        return self.focus_xFWHM

    def set_focus_area(self):
        if self.focus_shape is 'square':
            self.focus_area = self.focus_xFWHM * self.focus_yFWHM
        else:
            # Both ellipse and circle have the same equation.
            self.focus_area = np.pi / 4 * self.focus_xFWHM * self.focus_yFWHM

    def get_focus_area(self):
        return self.focus_area

    def set_photons_per_pulse(self, x):
        """
        :param x: photons per pulse
        """
        self.n_phot = x
        self.update()

    def get_photons_per_pulse(self):
        return self.n_phot

    def set_photons_per_pulse_per_area(self):
        self.phi_in = self.n_phot / self.focus_area

    def get_photons_per_pulse_per_area(self):
        return self.phi_in

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
        self.update()
