## LCLS developers
Test your code in 3 environments before pull request: 1) python2, 2) python3, and 3) python3 with cupy.  

For python2 envrionment:  
conda activate ana-1.5.11 (or the latest version)

For python3 environment:  
git clone https://github.com/slac-lcls/lcls2.git
source lcls2/setup_env.sh
export PYTHONPATH=<YOUR_PYSINGFEL_PATH>:$PYTHONPATH

To turn on cupy in pysingfel:  
export USE_CUPY=1

## Contributing code

Test code with Flake8 and Pytest before pull request

## Unit Test

With your environment loaded, run the unit tests with
```
cd pysingfel
# stop the build if there are Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
# unit test
pytest
```
from the root directory of this repository or from the pysingfel subdirectory.
If pysingfel is not installed in the environment and pytest cannot find it, add it to Python path by running
```
export PYTHONPATH=$PWD:$PYTHONPATH
```
from the root directory of the repository.
