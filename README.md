This branch was created by [Bardiya Akhbari](https://www.researchgate.net/profile/Bardiya_Akhbari) for additional debugging and implementing support for Particle Swarm Optimization (PSO) algorithm. Currently, the code has been optimized for CUDA, and particle swarm optimization method has been added to the registration options. Autoscoper 2 has been developed by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff). In order to simplify installation and compilation the build system has been changed to CMake and the UI has been switched to QT5. Version 2 has improved processing, several bugfixes and new functionality, e.g. multibone, batch processing, when compared to the original versions.


# Installer for Autoscoper v2.7 #
The installer for Autoscoper 2.7.0 can be find [on SimTk website](https://simtk.org/projects/autoscoper).

You need to install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?), and update your graphics card driver to run the application.
# Compiling Instructions #

## WINDOWS ##

Prerequisites

- [CUDA toolkit v10.1 or later](https://developer.nvidia.com/cuda-downloads?)
- [CMake 3.12 or later](https://cmake.org/)
- [Sourcetree](https://www.sourcetreeapp.com) or [git](https://git-scm.com/downloads)
- [QT 5.10 or later](https://www.qt.io/download)
- Update your graphics card driver

Build

1. Clone the [bitbucket repository](https://bitbucket.org/xromm/autoscoper-v2/src/BA_Playground/).
2. Run CMake and choose a source and the build folder for Autoscoper and click configure.
	1. On Windows choose a 64bit build of the Visual Studio version you have installed.
	2. Dependencies to tiff and glew will be installed automatically and the other dependencies should be found automatically.
	3. By default it will use CUDA.
	4. You receive an error for Qt5_DIR. Select the following path $QT_ROOT_PATH\msvc2017_64\lib\CMake\Qt5 (e.g., C:\Qt5\5.10.1\msvc2017_64\lib\cmake\Qt5).
3. Click configure again.	
4. Click generate
5. Click open project and build in Visual Studio.
6. To install build the INSTALL project in VisualStudio. This will build Autoscoper and installs it in your build folder in the sub-folder install/bin/Debug or install/bin/Release depending on which build was performed.

NOTE: Debugging a CUDA program is not straightforward in Visual Studio, so you cannot do the debugging similar to other applications.

## LINUX ##

1. Clone the [bitbucket repository](https://bitbucket.org/xromm/autoscoper-v2/src/BA_Playground/).
2. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate) 
3. Build using 'make'


# History of Autoscoper #
Autoscoper 1 was developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper).

Autoscoper was revamped and upgraded to version 2 by Dr. Ben Knorlein. Multi-bone tracking feature was added and a socket was design for interaction with Matlab.

Currently, Bardiya Akhbari maintains the software and debugs the reported issues.