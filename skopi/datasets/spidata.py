"""SPI single/multiple hit dataset"""
import os
import gdown
import tarfile
import numpy as np

def load_data(path=".tmp_1A8J6aiIsl1To8E7DtrysRkJfGVOQrXmX"):
    """Loads 3IYF dataset.
    This is a dataset of 30,000 128x128 diffraction images of chaperone (PDB:3iyf),
    along with a test set of 10,000 images. Each image contains single, double, triple,
    and quadruple particles. 
    Args:
      path: path where to cache the dataset locally
        (relative to $PWD).
    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
    **x_train**: float32 NumPy array of grayscale image data with shapes
      `(30000, 1, 128, 128)`, containing the training data.
    **y_train**: uint8 NumPy array of digit labels (integers in range 1-4)
      with shape `(30000,)` for the training data.
    **x_test**: uint8 NumPy array of grayscale image data with shapes
      (10000, 28, 28), containing the test data. Pixel values range
      from 0 to 255.
    **y_test**: uint8 NumPy array of digit labels (integers in range 1-4)
      with shape `(10000,)` for the test data.
    Example:
    ```python
    (x_train, y_train), (x_test, y_test) = skopi.datasets.3iyf.load_data()
    assert x_train.shape == (30000, 128, 128)
    assert x_test.shape == (10000, 128, 128)
    assert y_train.shape == (30000,)
    assert y_test.shape == (10000,)
    ```
    """
    fid = '1S2dFW__HJIAwjsiST5cF0C0bzBr1rbPg'
    url = 'https://drive.google.com/uc?id='+fid
    tarFile = os.path.join(path, 'spi_1A8J6aiIsl1To8E7DtrysRkJfGVOQrXmX.tar.gz')
    npzFile = os.path.join(path, 'spi_1A8J6aiIsl1To8E7DtrysRkJfGVOQrXmX.npz')
    createdTmp = False

    def cleanup():
        if os.path.exists(tarFile):
            os.remove(tarFile)
        if os.path.exists(npzFile):
            os.remove(npzFile)
        if os.path.exists(path) and createdTmp:
            os.rmdir(path)

    try:
        # download tarball from google drive
        if not os.path.exists(path):
            os.mkdir(path)
            createdTmp = True
        gdown.download(url, tarFile, quiet=False)

        # extract spi.npz
        tar = tarfile.open(tarFile)
        tar.extractall(path)
        tar.close()

        # load in memory
        f = np.load(npzFile, allow_pickle=True)
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)
    except PermissionError:
        print("Error: try again by providing a path argument with write-permission!")
    finally:
        cleanup()

    
