Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

The `dex-net` package can be installed as a standalone Python module or a ROS package.
We suggest installing as a standalone Python module unless you want to use `dex-net` to attempt to plan grasps for a physical robot.


Installation has been tested on Ubuntu 12.04 LTS, Ubuntu 14.04 LTS, and Ubuntu 16.04 LTS. We do not recommend using `dex-net` with anaconda, as there are  known issues with the installation of the visualization tools in conda envrionments.

1. Clone the repository
"""""""""""""""""""""""
Change directories to the desired install location of `dex-net`. If you are installing as a ROS package, this should be /path/to/your/catkin_ws

Clone or download the project from `Github`_. ::

  $ git clone https://github.com/BerkeleyAutomation/dex-net.git

.. _Github: https://github.com/BerkeleyAutomation/dex-net

2. Run the installation script
""""""""""""""""""""""""""""""
Run the `dex-net` installation helper script::

  $ sudo sh install.sh {cpu|gpu} {python|ros}

The brackets indicate optional arguments to switch installation methods.

The first argument specifies the version:

* **cpu:** no TensorFlow GPU support
* **gpu:** TensorFlow GPU support for GQ-CNN training

The second argument specifies the installation mode:

* **python:** Python-only installation. No ROS services will be installed for `dex-net` or any of the Berkeley AUTOLAB modules.
* **ros:** Installation as a ROS package. Enables ROS services for the Berkeley AUTOLAB modules and supports `dex-net` ROS nodes and services that may be developed in the future.

3. Test the installation
""""""""""""""""""""""""
To test your installation, run ::

    $ python setup.py test

We highly recommend testing before using the module.

4. Try it out!
""""""""""""""
Go the `dex-net` `Command Line Interface (CLI) example`_ to see the basic functionality.

.. _Command Line Interface (CLI) example: http://bit.ly/2uPEliy

Issues
~~~~~~
If you are having issues with the installation script, then you should try to install manually by following our `extended installation instructions`_, which include individual commands, optional dependencies, and workarounds for some known issues.

.. _extended installation instructions: https://docs.google.com/document/d/1YImq1cBTy9E1n1On6-00gueDT4hfmYJK4uOcxZIzPoY/edit?usp=sharing

Please raise installation issues on the `Github Issues`_.

.. _Github Issues: https://github.com/BerkeleyAutomation/dex-net/issues

Documentation
~~~~~~~~~~~~~

Building
""""""""
The API documentation is available on the `dex-net` `website`_.

.. _website: https://berkeleyautomation.github.io/dex-net/code.html

You can build `dex-net`'s documentation from scratch with a few extra dependencies --
specifically, `sphinx`_ and a few plugins. This is important for developers only.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

Go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying Documentation
"""""""""""""""""""""""
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.

