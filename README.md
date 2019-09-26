This is autoscoper 2 by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff). This branch was created by [Bardiya Akhbari](https://www.researchgate.net/profile/Bardiya_Akhbari) for additional debugging. The code has been optimized for CUDA, and particle swarm optimization method has been added to the registration options. The new Version 2 combines the sources of both the CUDA and OpenCL version and allows usage of either one. Furthermore in order to simplify installation and compilation the build system has been changed to CMake and the UI was switched to QT4. In addition, Version 2 has improved processing, several bugfixes and new functionality, e.g. multibone, batch processing, when compared to the original versions.

If you don't want to compile the code, you can download the installer for Autoscoper 2.7.0 here: https://brownbox.brown.edu/download.php?hash=5001f75e	

You need to install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?) before installing the software.
# Compiling Instructions #

## WINDOWS ##

Prerequisites

- CUDA toolkit v10.1 or later (https://developer.nvidia.com/cuda-downloads?)
- Cmake (https://cmake.org/)
- git (https://git-scm.com/downloads) or Sourcetree (https://www.sourcetreeapp.com)
- QT 5.10 or later
- Updated your graphics card driver

Build

1. Clone the git repository (https://bitbucket.org/xromm/autoscoper-v2.git)
2. Run cmake and choose a source and the build folder for autoscoper and click configure. On Windows choose a 64bit build of the Visual Studio version you have installed. Dependencies to tiff and glew will be installed automatically and the other dependencies should be found automatically. By default it will use OpenCL if you want to use CUDA instead set BUILD_WITH_CUDA. 
3. click generate
4. click open project and build in Visual studio.
5. to install build the INSTALL project in VisualStudio. This will build autoscoper and installs it in your build folder in the subfolder install/bin/Debug or install/bin/Release depending on which build was performed.

## LINUX ##

1. Install your GPGPU SDK. Either OpenCL or CUDA.
2. Install libtiff, glew, qt4 using your package manager
3. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate) 
4. Build using 'make'


# History of Autoscoper #
Autoscoper 1 was developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper).

Autoscoper was revamped and upgraded to version 2 by Dr. Ben Knorlein. Multibone tracking feature was added and a socket was design for interaction with Matlab.

Currently, Bardiya Akhbari maintains the software and debugs the reported issues.