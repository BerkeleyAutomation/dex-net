# Berkeley AUTOLAB's Dex-Net Package

## Links
[Documentation](https://berkeleyautomation.github.io/dex-net/code.html)

[Project website](https://berkeleyautomation.github.io/dex-net)

[RSS Paper](https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf)

## Overview
The dex-net Python package is for opening, reading, and writing HDF5 databases of 3D object models, parallel-jaw grasps, and grasp robustness metrics.

The HDF5 databases can also be used to generate massive datasets associating tuples of point clouds and grasps with binary grasp robustness labels to train [Grasp Quality Convolutional Neural Networks (GQ-CNNs)](https://berkeleyautomation.github.io/gqcnn) to predict robustness of candidate grasps from point clouds.
If you are interested in this functionality, please email Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in GQ-CNN Dataset Generation."

This package is part of the [Dexterity Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net) project.
Created and maintained by the [AUTOLAB at UC Berkeley](https://autolab.berkeley.edu).

## Installation
See [the documentation](https://berkeleyautomation.github.io/dex-net/code.html) for installation instructions and API Documentation.

## Datasets
The Dex-Net Object Mesh Dataset v1.1 and Dex-Net 2.0 HDF5 database can be downloaded from [the data repository](http://bit.ly/2uh07i9).

## Parallel-Jaw Gripper
The repository currently supports our custom ABB YuMi gripper and the Baxter gripper.
If you are interested in additional parallel-jaw grippers, please email Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in Contributing to the Dex-Net Grippers" with a description of the parallel-jaw gripper you'd like to add.

