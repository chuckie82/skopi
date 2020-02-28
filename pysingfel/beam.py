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

    def __init__(self, fname=None, **kwargs):
        """
        :param fname: The beam profile
        """
        if fname and kwargs:
            raise TypeError("Beam cannot accept fname and other arguments.")
        elif fname:
            self.read_beamfile(fname)
        else:
            self.init_from_arg_dict(kwargs)

        # Default polarization angle, requires input from user or file in the future
        self.Polarization = np.array([1, 0, 0])

    def init_from_arg_dict(self, arg_dict):
        self.set_wave_parameters_from_arg_dict(arg_dict)
        self.set_focus_from_arg_dict(arg_dict)
        try:
            fluence = arg_dict.pop("fluence")
        except KeyError:
            raise TypeError("Beam requires fluence argument.")
        self.set_photons_per_pulse(fluence)
        if arg_dict:
            raise TypeError("Non-understood Beam arguments: "
                            "{}".format(arg_dict))

    def set_wave_parameters_from_arg_dict(self, arg_dict):
        wave_arguments = {"wavelength", "photon_energy", "wavenumber"}
        given_wave_arguments = wave_arguments.intersection(arg_dict)
        number_of_wave_arguments = len(given_wave_arguments)

        if number_of_wave_arguments > 1:
            raise TypeError("Beam can only accept a single wavelength, "
                            "photon_energy, or wavenumber.")
        elif number_of_wave_arguments == 0:
            raise TypeError("Beam needs a wavelength, "
                            "photon_energy, or wavenumber.")

        for wave_argument in given_wave_arguments:
            # There should only be one left.
            # Remove used argument from dict.
            setattr(self, wave_argument, arg_dict.pop(wave_argument))

    def set_focus_from_arg_dict(self, arg_dict):
        if "focus_radius" in arg_dict:
            if {"focus_x", "focus_y"}.intersection(arg_dict):
                raise TypeError("Focus radius and focus x and y are not "
                                "compatible.")
            if arg_dict.get("focus_shape") not in (None, "circle"):
                raise TypeError("Focus radius only compatible with circle "
                                "shape.")
            arg_dict["focus_shape"] = "circle"
            # Rename and get rid of radius.
            arg_dict["focus_x"] = arg_dict.pop("focus_radius")
        # No more radius. Only x, y, and/or shape.

        if not "focus_x" in arg_dict:
            raise TypeError("Focus requires at minimum x dimension or radius.")
        focus_x = arg_dict.pop("focus_x")

        if not "focus_shape" in arg_dict:
            if "focus_y" in arg_dict:
                arg_dict["focus_shape"] = "ellipse"
            else:
                arg_dict["focus_shape"] = "circle"
        focus_shape = arg_dict.pop("focus_shape")

        if focus_shape not in {'circle', 'ellipse', 'square', 'rectangle'}:
            raise ValueError("Beam focus can only be circle, ellipse,"
                             "square, rectangle.")
        self._focus_shape = focus_shape

        if focus_shape in {'circle', 'square'}:
            if "focus_y" in arg_dict:
                raise TypeError("Focus with {} shape incompatible with "
                                "focus y.".format(focus_shape))
            self._focus_xFWHM = focus_x
            self._focus_yFWHM = focus_x
        elif focus_shape in {'ellipse', 'rectange'}:
            if "focus_y" in arg_dict:
                raise TypeError("Focus with {} shape requires "
                                "focus y.".format(focus_shape))
            self._focus_xFWHM = focus_x
            self._focus_yFWHM = focus_y
        self.set_focus_area()

    def set_focus(self, *args, **kwargs):
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
        arg_dict = {}
        if len(args) > 0:
            arg_dict['focus_x'] = args[0]
        if len(args) > 1:
            arg_dict['focus_y'] = args[1]
        if len(args) > 2:
            arg_dict['focus_shape'] = args[2]
        if len(args) > 3:
            raise TypeError("Too may arguments for set_focus.")

        for key, value in kwargs:
            new_key = key
            if not key.startswith("focus_"):
                new_key = "focus_" + key
            arg_dict[new_key] = kwargs[key]

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
                        self.set_photons_per_pulse(float(tmp[1]))
                    if tmp[0] == 'beam/radius':
                        self.set_focus(radius=float(tmp[1]))

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
