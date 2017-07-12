"""
Setup of Dex-Net python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'cvxopt',
    'dill',
    'h5py'
]

setup(name='dex-net',
      version='0.1.dev0',
      description='Dex-Net project code',
      author='Jeff Mahler',
      author_email='jmahler@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['dexnet'],
      install_requires=requirements,
      test_suite='test'
     )
