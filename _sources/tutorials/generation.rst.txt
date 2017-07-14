Point Cloud Dataset Generation
------------------------------
Robust grasping policies based on `Grasp Quality Convolutional Neural Networks`_ (GQ-CNNs) may be useful for planning grasps on novel objects with a physical robot.
This requires training GQ-CNNs on a dataset of synthetic point clouds, grasps, and grasp robustness metrics such as the `Dex-Net 2.0 dataset`_.

.. _Grasp Quality Convolutional Neural Networks: https://berkeleyautomation.github.io/gqcnn
.. _Dex-Net 2.0 dataset: http://bit.ly/2rIM7Jk

It may be beneficial to train GQ-CNNs on custom robot grippers and datasets of 3D object models based on the application.
We are working toward making this functionality publicly available.
If you are interested in this, please email Jeff Mahler (jmahler@berkeley.edu) wit\
h the subject line: "Interested in GQ-CNN Dataset Generation."
