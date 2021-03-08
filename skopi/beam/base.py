import numpy as np

from . import convert


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

        # variable to track original fluence if adding jitter
        self._n_phot_ideal = None

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
            # Note: focus_x corresponds to diameter.
            arg_dict["focus_x"] = 2 * arg_dict.pop("focus_radius")
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
                focus_y = arg_dict.pop("focus_y")
                if focus_x != focus_y:
                    # Complain if incompatible x and y
                    raise TypeError("Focus with {} shape incompatible with "
                                    "focus y.".format(focus_shape))
            self._focus_xFWHM = focus_x
            self._focus_yFWHM = focus_x
        elif focus_shape in {'ellipse', 'rectangle'}:
            if not "focus_y" in arg_dict:
                raise TypeError("Focus with {} shape requires "
                                "focus y.".format(focus_shape))
            focus_y = arg_dict.pop("focus_y")
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

        for key, value in kwargs.items():
            new_key = key
            if not key.startswith("focus_"):
                new_key = "focus_" + key
            arg_dict[new_key] = kwargs[key]

        self.set_focus_from_arg_dict(arg_dict)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self._photon_energy = convert.wavelength_to_photon_energy(
            self._wavelength)
        self._wavenumber = convert.wavelength_to_wavenumber(self._wavelength)

    @property
    def photon_energy(self):
        return self._photon_energy

    @photon_energy.setter
    def photon_energy(self, value):
        self._photon_energy = value
        self._wavelength = convert.photon_energy_to_wavelength(
            self._photon_energy)
        self._wavenumber = convert.wavelength_to_wavenumber(self._wavelength)

    @property
    def wavenumber(self):
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, value):
        self._wavenumber = value
        self._wavelength = convert.wavenumber_to_wavelength(self._wavenumber)
        self._photon_energy = convert.wavelength_to_photon_energy(
            self._wavelength)

    def get_wavevector(self):
        """
        Get the wave vector. Notice that here, the wavevector is defined as
        [0, 0, 1 / wavelength]
        :return: wavevector
        """
        return np.array([0, 0, 1.0 / self.wavelength])

    def get_focus(self):
        """
        Return focus parameters x, y, and shape.

        Lenghts x and y are FWHM. Shape is a string.
        :return: (x, y, shape)
        """
        return (self._focus_xFWHM, self._focus_yFWHM, self._focus_shape)

    def set_focus_area(self):
        if self._focus_shape in ('square', 'rectangle'):
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
                    if tmp[0] == 'beam/photonsPerShot':
                        self.set_photons_per_pulse(float(tmp[1]))
                    if tmp[0] == 'beam/radius':
                        self.set_focus(radius=float(tmp[1]))

    def get_highest_wavenumber_beam(self):
        """
        For variable/polychromatic beam to return highest wavenumber.
        """
        # If simple Beam, return itself.
        # Composed beams should return simple one.
        return self

    def generate_new_state(self):
        """
        Return list of Beams at each pulse to represent a variable spectrum.
        """
        # If simple Beam, return itself.
        # Variable beams should return simple one.
        return [self]

    def add_fluence_jitter(self, sigma):
        """
        Add jitter to maximum fluence, assuming variation is Gaussian.
        :param sigma: standard deviation of Gaussian in pixels
        :return fluence: adjusted fluence due to jitter
        """
        # track or reset to ideal fluence (self._n_phot, before jitter)
        if self._n_phot_ideal is None:
            self._n_phot_ideal = self._n_phot
        else:
            self._n_phot = self._n_phot_ideal

        # add fluence jitter
        fluence = np.random.normal(loc=self._n_phot, scale=sigma*self._n_phot, size=1)[0]
        self.set_photons_per_pulse(fluence)
        return fluence

    def fluence_at_position(self, position):
        """
        Compute flux at the particle's position, modeling the fluence within
        the beam's focus area as a 2d Gaussian in the plane of the beam and 
        constant along the direction of the beam (z-axis). Sigma is based on
        the given FWHM, and the fluence is assumed constant across the particle.
    
        :param position: particle's position in meters, origin is beam center
        :return fluence: fluence at particle's position
        """
        if self._n_phot_ideal is None:
            self._n_phot_ideal = self._n_phot
        else:
            self._n_phot = self._n_phot_ideal

        sigma = np.mean(np.array(self.get_focus()[:2])) / (2*np.sqrt(2*np.log(2)))
        p_radius = np.sqrt(np.sum(np.square(position[:2])))
        scale_factor = np.exp(-np.square(p_radius)/(2*np.square(sigma)))
        fluence = scale_factor*self._n_phot

        self.set_photons_per_pulse(fluence)    
        return fluence


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
