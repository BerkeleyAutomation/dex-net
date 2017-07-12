.. Dex-Net documentation master file, created by
   sphinx-quickstart on Thu Oct 20 10:31:18 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

database
========
Classes for access and storage of 3D models, images, grasps, and grasp quality metrics from Dex-Net.

Hdf5Database
~~~~~~~~~~~~
A wrapper for opening h5py databases that store datasets of 3D object models, parallel-jaw grasps, and grasp metrics.

.. autoclass:: dexnet.database.Hdf5Database

Hdf5Dataset
~~~~~~~~~~~~
A wrapper for h5py functions to read and write individual datasets of 3D object models, parallel-jaw grasps, and grasp metrics.
The intended use is to have a separate dataset for each unique source of 3D object models (e.g. 3DNet, KIT).

.. autoclass:: dexnet.database.Hdf5Dataset

Hdf5ObjectFactory
~~~~~~~~~~~~~~~~~
Low-level conversion methods between h5py groups and datasets to common object types.

.. autoclass:: dexnet.database.Hdf5ObjectFactory

MeshProcessor
~~~~~~~~~~~~~
Encapsulates mesh cleaning & preprocessing pipeline for database generation.

..autoclass:: dexnet.database.MeshProcessor
