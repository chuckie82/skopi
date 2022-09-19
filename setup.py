import setuptools
from io import open

requirements = [
    'numpy',
    'numba',
    'scipy',
    'mpi4py',
    'h5py',
    'six',
    'scikit-learn',
    'pytest',
    'matplotlib',
    'setuptools',
    'gdown',
    'tarfile'
]

setuptools.setup(name='skopi',
      maintainer='Chunhong Yoon',
      version='0.5.3',
      maintainer_email='yoon82@stanford.edu',
      description='Single particle imaging simulation package',
      long_description=open('README.md', encoding='utf8').read(),
      long_description_content_type="text/markdown",
      url='https://github.com/chuckie82/skopi.git',
      packages=setuptools.find_packages(),
      install_requires=requirements,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],
      zip_safe=False)
