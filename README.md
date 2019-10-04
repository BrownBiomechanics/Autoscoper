This is Autoscoper 2 by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff). This branch was created by [Bardiya Akhbari](https://www.researchgate.net/profile/Bardiya_Akhbari) for additional debugging. The code has been optimized for CUDA, and particle swarm optimization method has been added to the registration options. Furthermore in order to simplify installation and compilation the build system has been changed to CMake and the UI was switched to QT5. In addition, Version 2 has improved processing, several bugfixes and new functionality, e.g. multibone, batch processing, when compared to the original versions.

If you don't want to compile the code, you can download [the installer for Autoscoper 2.7.0 here](https://brownbox.brown.edu/download.php?hash=342e8dd5).

You need to install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?) before installing the software.
# Compiling Instructions #

## WINDOWS ##

Prerequisites

- CUDA toolkit v10.1 or later (https://developer.nvidia.com/cuda-downloads?)
- CMake 3.12 or later (https://cmake.org/)
- Sourcetree (https://www.sourcetreeapp.com) or git (https://git-scm.com/downloads)
- QT 5.10 or later (https://www.qt.io/download)
- Update your graphics card driver

Build

1. Clone the bitbucket repository (https://bitbucket.org/xromm/autoscoper-v2.git)
2. Run CMake and choose a source and the build folder for Autoscoper and click configure.
	2.1. On Windows choose a 64bit build of the Visual Studio version you have installed.
	2.2. Dependencies to tiff and glew will be installed automatically and the other dependencies should be found automatically.
	2.3. By default it will use CUDA.
	2.4. You receive an error for Qt5_DIR. Select the following path $QT_ROOT_PATH\msvc2017_64\lib\CMake\Qt5 (e.g., C:\Qt5\5.10.1\msvc2017_64\lib\cmake\Qt5).
3. Click configure again.	
4. Click generate
5. Click open project and build in Visual Studio.
6. To install build the INSTALL project in VisualStudio. This will build Autoscoper and installs it in your build folder in the sub-folder install/bin/Debug or install/bin/Release depending on which build was performed.

NOTE: Debugging a CUDA program is not straightforward in Visual Studio, so you cannot do the debugging similar to other applications.

## LINUX ##

1. Install your GPGPU SDK. Either OpenCL or CUDA.
2. Install libtiff, glew, qt4 using your package manager
3. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate) 
4. Build using 'make'


# History of Autoscoper #
Autoscoper 1 was developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper).

Autoscoper was revamped and upgraded to version 2 by Dr. Ben Knorlein. Multi-bone tracking feature was added and a socket was design for interaction with Matlab.

Currently, Bardiya Akhbari maintains the software and debugs the reported issues.