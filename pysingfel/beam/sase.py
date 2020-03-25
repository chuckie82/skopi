import numpy as np

from .base import Beam


class SASEBeam(Beam):
    def __init__(self, bandwidth=None, spike_width=None, n_spikes=0,
                 *args, **kargs):
        super(SASEBeam, self).__init__(**kargs)
        self.bandwidth = bandwidth
        self.spike_width = spike_width
        self.n_spikes = n_spikes

    def get_highest_wavenumber_beam(self):
        """
        For variable/polychromatic beam to return highest wavenumber.
        """
        return Beam(
            wavenumber=self.wavenumber * (1+0.5*self.bandwidth),
            focus_x=self._focus_xFWHM,
            focus_y=self._focus_yFWHM,
            focus_shape=self._focus_shape,
            fluence=self.get_photons_per_pulse()
        )

    def generate_new_state(self):
        """
        For variable beam to return specific instance.
        """
        # If simple Beam, return itself.
        # Variable beams should return simple one.
        wavenumbers = [
            self.wavenumber * (1 + (np.random.random()-0.5)*self.bandwidth)
            for i in range(self.n_spikes)
        ]
        fluences = (self.get_photons_per_pulse()
                    * np.random.dirichlet(np.ones(self.n_spikes)))
        return [
            Beam(
                wavenumber=wavenumbers[i],
                focus_x=self._focus_xFWHM,
                focus_y=self._focus_yFWHM,
                focus_shape=self._focus_shape,
                fluence=fluences[i])
            for i in range(self.n_spikes)
        ]
