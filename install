#!/bin/sh

# read cmd line inputs
VERSION=$1 # cpu or gpu
MODE=$2 # python or ros

# set cpu/gpu conditional libraries
case "${VERSION}"
in
cpu)
	TENSORFLOW_LIB=tensorflow
	;;
gpu)
	TENSORFLOW_LIB=tensorflow-gpu
	;;
*)
	echo "Usage: $0 {cpu|gpu} {python|ros}"
	exit 1
esac

echo "Installing Dex-Net in ${MODE} mode with ${VERSION} support"

# set workspace
case "${MODE}"
in
python)
	MODULES_DIR=deps # installs modules in deps folder
	;;
ros)
	MODULES_DIR=../ # installs in catkin workspace
	;;
*)
	echo "Usage: $0 {cpu|gpu} {python|ros}"
	exit 1
esac

# install apt deps
sudo apt-get install cmake libvtk5-dev python-vtk python-sip python-qt4 libosmesa6-dev meshlab libhdf5-dev

# install pip deps
pip install numpy scipy scikit-learn scikit-image opencv-python pyassimp tensorflow h5py mayavi matplotlib catkin_pkg multiprocess dill cvxopt ipython pillow pyhull setproctitle trimesh

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

# install autolab modules
cd ${MODULES_DIR}
git clone https://github.com/BerkeleyAutomation/autolab_core.git
git clone https://github.com/BerkeleyAutomation/perception.git
git clone https://github.com/BerkeleyAutomation/gqcnn.git
git clone https://github.com/BerkeleyAutomation/meshpy.git
git clone https://github.com/BerkeleyAutomation/visualization.git

# install meshpy
cd meshpy
python setup.py develop
cd ../

# install all Berkeley AUTOLAB modules
case "${MODE}"
in
python)
	# autolab_core
	cd autolab_core
	python setup.py develop
	cd ..

	# perception
	cd perception
	python setup.py develop
	cd ..

	# gqcnn
	cd gqcnn
	python setup.py develop
	cd ..

	# visualization
	cd visualization
	python setup.py develop
	cd ..
	cd ..
	;;
ros)
	# catkin
	cd ..
	catkin_make
	source devel/setup.bash
	cd src/dex-net
	;;
*)
	echo "Usage: $0 {cpu|gpu} {python|ros}"
	exit 1
esac

# install dex-net
python setup.py develop
