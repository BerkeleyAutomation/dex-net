Download Link
~~~~~~~~~~~~~
Dex-Net HDF5 databases and datasets of 3D object models are available from our `data repository`_.
For GQ-CNN training datasets, see the `gqcnn data repository`_
New datasets and 3D models will be uploaded to this location as they become available.

.. _data repository: http://bit.ly/2uh07i9
.. _gqcnn data repository: https://berkeley.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb

Dex-Net Object Mesh Dataset v1.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `Dex-Net Object Mesh Dataset v1.1`_ contains 1,500 3D object models in .OBJ format.
The source of the models are: 1,371 synthetic models from the 50 category subset of `3DNet`_ and 129 laser-scanned models from the `KIT Object Database`_.
The models were rescaled, rotated to align with the principal axes of the object, and centered on the center of hte object bounding box. 

More details can be found in the `Dex-Net 2.0 paper`_.

.. _Dex-Net Object Mesh Dataset v1.1 : http://bit.ly/2tLnRrQ
.. _3DNet: https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/
.. _KIT Object Database: https://h2t-projects.webarchiv.kit.edu/Projects/ObjectModelsWebUI/
.. _Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf

Dex-Net 2.0 HDF5 Database
~~~~~~~~~~~~~~~~~~~~~~~~~
The `Dex-Net 2.0 HDF5 Database`_ contains the Dex-Net Object Mesh Dataset v1.1 stored in HDF5 format. Each 3D model is labeled with up to 100 parallel-jaw grasps specified as an end-effector pose for an ABB YuMi custom gripper. Each grasp is labeled with force closure and the robust epsilon metric (also known as robust Ferrari-Canny) under uncertainty in object pose, gripper pose, and friction.

.. _Dex-Net 2.0 HDF5 Database : http://bit.ly/2vb3OCz

License
~~~~~~~
The datasets include 3D object mesh models from `3DNet`_ and the `KIT Object Database`_ that may be subject to copyright.
Please see the original datasets for more information.

.. _3DNet: https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/
.. _KIT Object Database: https://h2t-projects.webarchiv.kit.edu/Projects/ObjectModelsWebUI/

The grasps and grasp metrics are available for unrestricted use.
