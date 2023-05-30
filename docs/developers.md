## LCLS developers
Test your code in 3 environments before pull request: 1) python2, 2) python3, and 3) python3 with cupy.  

For python2 envrionment:  
```
source /reg/g/psdm/etc/psconda.sh
```

For python3 environment:  
```
git clone https://github.com/slac-lcls/lcls2.git
source lcls2/setup_env.sh
export PYTHONPATH=<YOUR_PYSINGFEL_PATH>:$PYTHONPATH
```

To turn on cupy in skopi:  
```
export USE_CUPY=1
```

## Contributing jupyter notebooks
Add notebooks to skopi/examples/notebooks and add a description in skopi/docs/tutorials.md

## Contributing code

Test code with Flake8 and Pytest before pull request

## Downloading test data
Prior to running pytest, use following script to download and extract test data into the right path.
```
cd skopi/examples/input
bash ./download.sh
tar -xf lcls.tar.gz
```
Your test data (0-end.data) should be in this path: `skopi/examples/input/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data`.

## Unit Test

With your environment loaded, run the unit tests with
```
cd skopi
# stop the build if there are Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
# unit test
pytest
```
from the root directory of this repository or from the skopi subdirectory.
If skopi is not installed in the environment and pytest cannot find it, add it to Python path by running
```
export PYTHONPATH=$PWD:$PYTHONPATH
```
from the root directory of the repository.
