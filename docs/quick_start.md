## Getting Started for LCLS simulations

Start by downloading a copy of pysingfel package and LCLS calibration
```
git clone https://github.com/chuckie82/pysingfel.git
cd pysingfel/examples/input && source download.sh
tar -xvf lcls.tar.gz
```

## Example

Simulate open and closed states of chaperones on an LCLS pnCCD detector
```
cd pysingfel/examples/scripts
python ExampleMultipleChaperones.py
```
