#!/usr/bin/env python
# coding: utf-8

########## SASE Spectrum ###############
# The random nature of the SASE spectrum makes it difficult to fit a distribution. In this example script, we demonstrate how to approximate the real spectrum by using kernel density estimation (KDE), a non-parametric way to estimate the probability density function of a random varaible, where inferences about the population are made based on a finite data sample. 
# Here, we provide a small subset of LS49 SASE spectra generated from nanoBragg: 3 pulses at 7120eV and 3 pulses at 3560eV, where the LS49 specifications can be found at https://pswww.slac.stanford.edu/questionnaire_slac/proposal_questionnaire/run16/LS49/#xray
# Users can change the value of the parameters to match with the specific SASE spectra for their experiments.

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import time, os
import skopi as sk

def wavelength_to_photon_energy(wavelength):
    """
    Convert wave length to photon energy in eV
    :param wavelength: wavelength in m.
    :return:
    """
    return 1.2398e-06 / wavelength

# Input files
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input')
beamfile=input_dir+'/beam/amo86615.beam'

# Load SASE spectra, with the mean photon energy = 7120eV
LS49_SASE_7120eV = np.load(input_dir+'/beam/LS49_SASE_7120eV.npz')
wavlen_7120eV = LS49_SASE_7120eV['wavelength']
flux_7120eV_ebeam0 = LS49_SASE_7120eV['flux_ebeam0']
flux_7120eV_ebeam1 = LS49_SASE_7120eV['flux_ebeam1']
flux_7120eV_ebeam2 = LS49_SASE_7120eV['flux_ebeam2']

# Load beam
mu = 7120. # mean photon energy
sigma = 10. # standard deviation
n_spikes = 100 # number of spikes in a SASE pulse
sase_beam = sk.SASEBeam(mu, sigma, n_spikes, fname=beamfile)
spikes = sase_beam.generate_new_state()
print('mean photon energy of the SASE beam = {} eV'.format(sase_beam.photon_energy))
print('photon energy of individual spikes = {} eV'.format([sp.photon_energy for sp in spikes]))

energy = []
flux = []
for i in range(n_spikes):
    energy.append(spikes[i].photon_energy)
    flux.append(spikes[i].get_photons_per_pulse())

samples = np.random.normal(mu, sigma, n_spikes)

# Kernel Density Estimation (KDE)
gkde = stats.gaussian_kde(samples)
ind = np.linspace(mu-50, mu+50, n_spikes+1)
gkde.set_bandwidth(bw_method='scott') # bw_method = scott, silverman, gkde.factor/3., gkde.factor/4., etc.
kdepdf = gkde(ind)

# Visualization
plt.vlines(wavelength_to_photon_energy(wavlen_7120eV)*10**10, 0, flux_7120eV_ebeam2/np.array(flux_7120eV_ebeam2).sum()*sase_beam.get_photons_per_pulse(), color="b", label="real SASE")
plt.vlines(energy, 0, flux, color="r", label="simulated SASE")
plt.plot(ind, kdepdf*sase_beam.get_photons_per_pulse(), color='g', label="KDE")
plt.xlabel('energy [eV]')
plt.ylabel('flux [# photons/pulse]')
plt.legend(loc="upper left")
plt.ylim(0, 1.5e11)
plt.show()


