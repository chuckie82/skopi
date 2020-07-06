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
prody          >1.10 (Required for conformations)


## Setting Up LCLS2 Conda Environment

1. Install LCLS2 on psana
Refer to README.md on the webpage https://github.com/slac-lcls/lcls2


2. Install CuPy on psana
```
git clone https://github.com/cupy/cupy.git
pip install --no-cache-dir .
```


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

## Setting Up Node Proxy for Psana2 calibration constants

1. Install

```
mkdir test_proxy
cd test_proxy
wget https://nodejs.org/dist/v12.16.3/node-v12.16.3-linux-ppc64le.tar.gz
tar xvfz node-v12.16.3-linux-ppc64le.tar.gz
ln -s node-v12.16.3-linux-ppc64le latest
wget https://raw.githubusercontent.com/ExaFEL/installation/master/proxy.js
wget https://raw.githubusercontent.com/ExaFEL/installation/master/run_proxy.sh
chmod +x run_proxy.sh
export PATH=$PWD/latest/bin:$PATH
npm --version
node --version
npm install http-proxy -save
```

2. Change proxy listen port -- use your own port number to avoid conflicting with other users.

In proxy.js, change 6749 to any other number.
```
var httpProxy = require('http-proxy');
httpProxy.createProxyServer({
  target: {
    protocol: 'https:',
    host: 'pswww.slac.stanford.edu',
    port: 443
  },
  changeOrigin: true,
}).listen(6749); 
```

3. Run the proxy script

```
./run_proxy.sh
```

4. Set environment variable for psana2

```
export LCLS_CALIB_HTTP=http://YOUR_HOST_NAME:LISTEN_PORT_NUMBER/calib_ws
```
where YOUR_HOST_NAME is your current node (type printenv HOSTNAME to see the name) and LISTEN_PORT_NUMBER is the number that you chose in step 2.
