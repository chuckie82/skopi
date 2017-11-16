from setuptools import setup
from io import open

requirements = [
            'numpy',
            'numba',
            'scipy',
            'mpi4py',
            'h5py'
                ]

setup(name='pysingfel',
      maintainer = 'Haoyuan Li',
      version = '0.1.1',
      maintainer_email = 'hyli16@stanford.edu',
      description='Python version of singfel.',
      long_description=open('README.rst', encoding='utf8').read(),
      url='https://github.com/Haoyuan-Li-93/pysingfel.git',
      packages=['pysingfel','pysingfel.gpu'],
      scripts=['bin/radiationDamageMPI'],
      install_requires=requirements,
      zip_safe=False)
