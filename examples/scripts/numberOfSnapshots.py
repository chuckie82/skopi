import numpy as np
import argparse

"""
Calculate the number of snapshots required for a 3D reconstruction.

For more detail, refer to and cite:
I. Poudyal, M. Schmidt, and P. Schwander, “Single-particle imaging by x-ray free-electron lasers—How many snapshots are needed?,” Struct. Dyn., vol. 7, no. 2, p. 024102, Mar. 2020.
"""

parse = argparse.ArgumentParser()
parse.add_argument('-r', '--res', type=float, help='desired resolution (Angstrom)')
parse.add_argument('-d', '--diameter', type=float, help='particle diameter (Angstrom)')
parse.add_argument('-n', '--nPhotons', type=float, help='number of photons per Shannon pixel at the desired resolution')
parse.add_argument('-s', '--snr', type=float, default=1.0, help='signal to noise ratio at the desired resolution shell')
parse.add_argument('-p', '--pObserve', type=float, default=0.5, help='desired joint probability to observe photons')
parse.add_argument('-m', '--maxSnapshots', type=int, default=1e6, help='maximum number of snapshots to search over')
args = parse.parse_args()

d = args.res
D = args.diameter
nPhotons = args.nPhotons
P_tilde = args.pObserve
SNR = args.snr
nSmax = args.maxSnapshots

def logFactorial(n):
    if n < 20:
        value = np.log(np.math.factorial(n))
    else:
        value = 0.5*np.log(2*np.pi*n) + n*np.log(n/np.e)
    return value

def pnm(p,N,M):
    """
    Probability to observe a voxel at least M times
    from an ensemble of N snapshots
    p = probability to hit a voxel for a single snapshot
    N = Number of Snapshots
    M = Redundancy
    """
    if N < M:
        s = 1
    else:
        s = 0
        lp = np.log(p)
        lmp = np.log(1-p)
        for k in np.arange(M):
            s = s + np.exp(logFactorial(N) - logFactorial(N-k) - logFactorial(k) + k*lp + (N-k)*lmp)
    return np.maximum(1-s,0)

def numberOfSnapShots(d, D, nPhotons, SNR, P_tilde):
    """
    Return number of snapshots required given the experimental condition
    d is the desired resolution
    D is the diameter of the particle
    nPhotons is the number of photons per Shannon pixel at the desired resolution
    P_tilde is the desired joint probability to observe photons
    Output:
    nS is number of snapshots required for 3D reconstruction
    """
    # Number of Resolution elements
    R = D/d
    # Number of voxels at Resolution Shell
    nV_Shell = 16*np.pi*R**2
    # Probability per Shannon voxel
    p = 1./(4*R)
    M = np.ceil(SNR**2/nPhotons)
    # P -> Probability to observe a voxel at least M times from an ensemble of nS snapshot
    # obtained from given P_tilde
    P = np.exp(2*np.log(P_tilde)/nV_Shell)
    step = 2**10
    nS0 = M
    while step > 1:
        for nS in np.arange(nS0,nSmax,step):
            prob = pnm(p,nS,M)
            if prob > P:
                break
        nS0 = nS - step
        step = step/2
    return nS

nS = numberOfSnapShots(d, D, nPhotons, SNR, P_tilde)
print("Number of snapshots required: ", nS)
