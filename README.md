# Pysingfel
Python-based Single Particle Imaging Simulation for Free-Electron Lasers

## Unit Test

Run unit tests
```
pytest
```

## Example

Simulate open and closed states of chaperones on an LCLS pnccd detector
```
python pysingfel/examples/scripts/ExampleMultipleChaperone.py
```

## Dependencies

python         [>2.6 >3.5]
numpy          >1.10
numba          >0.46
six            >1.2
scitkit-learn  >0.19
pytest         >4.5
matplotlib     >2.1
setuptools     >44.0
h5py           >2.6
scipy          >1.1
mpi4py         >=2.0

Optional:
psana-conda    >1.3
