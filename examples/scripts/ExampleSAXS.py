# SAXS simulation:
import skopi as sk

# Set up particle
# pdb file of lidless mmCpn in open state
particle = sk.Particle()
particle.read_pdb("../input/pdb/3iyf.pdb", ff='WK')

N = 100000 # no. of random HKL samples
resmax = 1e-9 # maximum resolution of the SAXS curve

saxs = sk.SAXS(particle,N,resmax)
saxs.plot()


