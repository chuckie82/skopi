## Getting Started for LCLS simulations

Start by downloading a copy of skopi package and LCLS calibration
```
git clone https://github.com/chuckie82/skopi.git
cd skopi/examples/input && source download.sh
tar -xvf lcls.tar.gz
```

## Example

Simulate open and closed states of chaperones on an LCLS pnCCD detector
```
cd skopi/examples/scripts
python ExampleMultipleChaperones.py
```
