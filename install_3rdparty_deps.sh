#!/bin/sh

# install apt deps
sudo apt-get install cmake libvtk5-dev python-vtk python-sip python-qt4 libosmesa6-dev meshlab libhdf5-dev

# install pip deps
pip install numpy scipy scikit-learn scikit-image opencv-python pyassimp tensorflow tensorflow-gpu h5py mayavi matplotlib catkin_pkg multiprocess dill cvxopt ipython pillow pyhull setproctitle trimesh

# install deps from source
mkdir deps
cd deps

# install SDFGen
git clone https://github.com/jeffmahler/SDFGen.git
cd SDFGen
sudo sh install.sh
cd ..

# install Boost.NumPy
git clone https://github.com/jeffmahler/Boost.NumPy.git
cd Boost.NumPy
sudo sh install.sh
cd ..

# return to dex-net directory
cd ..
