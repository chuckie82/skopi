import numpy as np


# several converters
def wavelength2wavenumber(Lambda):
    """
    Lambda is the wavelength in meter.
    Return 1./Lambda
    """
    return 1. / Lambda


def wavenumber2wavelength(k):
    """
    k is the wavenumber in meter^-1.  k= 1./wavelength.
    Return 1./k
    """
    return 1. / k


def photonEnergy2wavelength(photonEnergy):
    """
    The unit of photonEnergy is eV.
    The unit of wavelength is meter.
    Return 1.2398e-6 / photonEnergy
    """
    return 1.2398e-6 / photonEnergy


def wavelength2photonEnergy(wavelength):
    """
    The unit of photonEnergy is eV.
    The unit of wavelength is meter.
    Return 1.2398e-6 / wavelength
    """
    return 1.2398e-6 / wavelength


class Beam(object):
    def __init__(self, *fname):
        self.Lambda = 0  # (m) wavelength
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
            self.readBeamFile(fname[0])

    def update(self):
        """
        Update the related properties of the class object, after
        a certain property is provided.
        """
        if self.Lambda != 0:
            self.photon_energy = wavelength2photonEnergy(self.Lambda)
            self.k = wavelength2wavenumber(self.Lambda)
        elif self.photon_energy != 0:
            self.Lambda = photonEnergy2wavelength(self.photon_energy)
            self.k = wavelength2wavenumber(self.Lambda)
        elif self.k != 0:
            self.Lambda = wavenumber2wavelength(self.k)
            self.photon_energy = wavelength2photonEnergy(self.Lambda)

        if self.focus_xFWHM != 0:
            self.set_focusArea()
            if self.n_phot != 0:
                self.set_photonsPerPulsePerArea()

    # setters and getters
    def set_wavelength(self, x):
        """
        Set the wavelength to x.
        The unit of x is meter.
        """
        
        self.Lambda = x
        self.update()

    def get_wavelength(self):
        return self.Lambda

    def set_photon_energy(self, ev):
        """
        Set the photon_energy to eV.
        The unit of photon_energy is eV.
        """
        
        self.photon_energy = ev
        self.update()

    def get_photon_energy(self):
        return self.photon_energy

    # I am not sure whether there should be 2 pi or not
    def get_wavenumber(self):
        return self.k*2*np.pi
    
    def get_wavevector(self):
        return np.array([0,0,self.get_wavenumber()])

    def set_focus(self, x, y=None, shape='circle'):
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

    def set_focusArea(self):
        if self.focus_shape is 'square':
            self.focus_area = self.focus_xFWHM * self.focus_yFWHM
        else:
            # Both ellipse and circle have the same equation.
            self.focus_area = np.pi / 4 * self.focus_xFWHM * self.focus_yFWHM

    def get_focus_area(self):
        return self.focus_area

    def set_photonsPerPulse(self, x):
        self.n_phot = x
        self.update()

    def get_photonsPerPulse(self):
        return self.n_phot

    def set_photonsPerPulsePerArea(self):
        self.phi_in = self.n_phot / self.focus_area

    def get_photonsPerPulsePerArea(self):
        return self.phi_in

    def readBeamFile(self, fname):
        """
        Read Beam file and set the corresponding property.
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
