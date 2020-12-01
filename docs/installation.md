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
llvmlite
lmfit

Optional:  
psana-conda    >1.3  (Required for LCLS simulations only)  
prody          >1.10 (Required for conformations)


## Installing psana2

To install the LCLS2 code base, which includes psana2, refer to README.md on the webpage: https://github.com/slac-lcls/lcls2

## Installing pysingfel

To install pysingfel, run
```
python -m pip install -e pysingfel
```
or
```
pip install -e .
```
in the root directory.

## Installing and using CuPy

pysingfel uses GPU acceleration when computing the diffraction volumes using numba.cuda.
The slicing of these volumes into 2D diffraction patterns is then used on the CPU via NumPy.
These operations can be offloaded to the GPU as well using CuPy: https://cupy.chainer.org/.

To install CuPy, you might be able to clone it from GitHub
```
git clone https://github.com/cupy/cupy.git
```
then, from the cupy root directory and within your Python environment, install it with the following command:
```
pip install --no-cache-dir .
```
Alternatively, you can try
```
pip install cupy --no-cache-dir
```

Using CuPy changes the behavior of pysingfel in that it returns a CuPy array when you would be expecting a NumPy one, which other libraries might not be able to use.
You can also not use NumPy functions on CuPy arrays.
However, you can access NumPy/CuPy (whichever pysingfel uses) by using `xp` from the module `util`.
You can then write NumPy/CuPy compatible code using functions as `xp.abs(numpy_or_cupy_array)`.
Sometimes, you need to actually have a NumPy array. To cast either arrays into a NumPy one, use `asnumpy` from the same module: `asnumpy(numpy_or_cupy_array)`.
The reverse operation (turning a NumPy array into a NumPy/CuPy one) can be performed using `xp.asarray`.
To know which one you are using, type `xp.__name__`.
Get `xp` and `asnumpy` with:
```
from pysingfel.util import xp, asnumpy
```

Since CuPy can be difficult to use for untrained users, it is deactivated by default. pysingfel will only use CuPy if CuPy is installed and the environment variable USE_CUPY is set to 1 (before loading pysingfel).
To set the environment variable, type
```
export USE_CUPY=1
```
in your terminal. In a Jupyter session, run
```
%env USE_CUPY=1
```
in a cell.


## Installing prody

```
conda env create -f environment.yml
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
