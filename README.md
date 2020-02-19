# Pysingfel
Python-based Single Particle Imaging Simulation for Free-Electron Lasers

## Getting Started for LCLS simulations

Start by downloading a copy of pysingfel package and data folder  
```
git clone https://github.com/chuckie82/pysingfel.git  
cd pysingfel/examples/input  
Download a copy of lcls.tar.gz in the /input folder from https://stanford.box.com/s/e7c30tvhfz0485j2xr48rnrvel8a4yby
tar -xvf lcls.tar.gz
```

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
psana-conda    >1.3  (Required for LCLS simulations only)

## Quick-install on Summit

To quickly install a standalone version of pysingfel with psana2 on Summit, run
```
./setup/easy_install_summit_psana2.sh
```
from the root of the package.

This will:
  - downloads a fresh copy of Conda;
  - creates an environment with Python 3 and all the required packages for psana and pysingfel;
  - downloads and installs psana2; and
  - installs pysingfel.

To recover the environment, run `source setup/env.sh`.
