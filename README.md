# Berkeley AUTOLAB's Dex-Net Package

## Links
[Documentation](https://berkeleyautomation.github.io/dex-net/code.html)

[Project website](https://berkeleyautomation.github.io/dex-net)

[RSS Paper](https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf)

## Updates
As of Jan 1, 2018 the AUTOLAB visualization module uses the [trimesh](https://github.com/mikedh/trimesh) library instead of [meshpy](https://github.com/BerkeleyAutomation/meshpy).
Version mismatches between cloned libraries may lead to exceptions when using the CLI.
If you experience visualization errors, please run `git pull origin master` from the dex-net, meshpy, and visualization repositories and try again.

We are currently working on migrating dex-net to use [trimesh](https://github.com/mikedh/trimesh) and improving the installation procedure.
We hope to release a new version by May 2018.

## Overview
The dex-net Python package is for opening, reading, and writing HDF5 databases of 3D object models, parallel-jaw grasps, and grasp robustness metrics.

The HDF5 databases can also be used to generate massive datasets associating tuples of point clouds and grasps with binary grasp robustness labels to train [Grasp Quality Convolutional Neural Networks (GQ-CNNs)](https://berkeleyautomation.github.io/gqcnn) to predict robustness of candidate grasps from point clouds.
If you are interested in this functionality, please email Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in GQ-CNN Dataset Generation."

This package is part of the [Dexterity Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net) project.
Created and maintained by the [AUTOLAB at UC Berkeley](https://autolab.berkeley.edu).

## Installation
See [the documentation](https://berkeleyautomation.github.io/dex-net/code.html) for installation instructions and API Documentation.

## Usage
As of Feb. 1, 2018, the code is licensed according to the UC Berkeley Copyright and Disclaimer Notice.
The code is available for educational, research, and not-for-profit purposes (for full details, see LICENSE).
If you use this code in a publication, please cite:

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics." Robotics: Science and Systems (2017). Boston, MA.

## Datasets
The Dex-Net Object Mesh Dataset v1.1 and Dex-Net 2.0 HDF5 database can be downloaded from [the data repository](http://bit.ly/2uh07i9).

Custom datasets can now be generated using the script tools/generate_gqcnn_dataset.py

## Parallel-Jaw Grippers
The repository currently supports our custom ABB YuMi gripper.
If you are interested in additional parallel-jaw grippers, please email Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in Contributing to the Dex-Net Grippers" with a description of the parallel-jaw gripper you'd like to add.

## Custom Database Generation
The master Dex-Net API does not support the creation of new databases of objects. 
If you are interested in using this functionality for research, see [the custom-databases branch](https://github.com/BerkeleyAutomation/dex-net/tree/custom-databases).
However, we cannot provide support at this time.