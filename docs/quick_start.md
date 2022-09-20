## Getting Started on Google Colab
```python
!pip install skopi
import skopi
import matplotlib.pyplot as plt

# Load SPI dataset with single/double/triple/quadruple hits (PDB:3iyf)
(x_train,y_train), (x_test,y_test) = skopi.datasets.spidata.load_data()

# Display x_train images with y_train labels
plt.figure(figsize=(9,9))
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(x_train[i,0], vmax=3); plt.title(y_train[i]);
plt.show()
```
<div class="row">
  <div class="column">
    <p align="left"><img src="https://user-images.githubusercontent.com/1917664/191310322-ac051212-431f-4faf-91e2-2ed8b0711b69.png" alt="chaperone dataset with single/double/triple/quadruple hits (PDB:3iyf)" width="550px" height=auto></p>
  </div>
</div>

## Getting Started with LCLS simulations

Start by downloading a copy of skopi package and LCLS calibration
```
git clone https://github.com/chuckie82/skopi.git
cd skopi/examples/input && source download.sh
tar -xvf lcls.tar.gz
```
Simulate a chaperone on an LCLS pnCCD detector
```
cd skopi/examples/scripts
python ExampleSPI.py
```

<div class="row">
  <div class="column">
    <p align="left"><img src="https://user-images.githubusercontent.com/1917664/191306733-aef4655c-60bf-4a70-9daa-5c6d2defe746.png" alt="chaperone diffraction image (PDB:3iyf)" width="450px" height=auto></p>
  </div>
</div>
