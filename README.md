This is autoscoper 2 by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff). Autoscoper 1 was developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper). The new Version 2 combines the sources of both the CUDA and OpenCL version and allows usage of either one. Furthermore in order to simplify installation and compilation the build system has been changed to CMake and the UI was switched to QT4. In addition, Version 2 has improved processing, several bugfixes and new functionality, e.g. multibone, batch processing, when compared to the original versions.

Autoscoper either requires an OpenCL or a CUDA SDK installed. You can set the desired GPGPU Framework by using the cmake Flag "BUILD_WITH_CUDA". If not set OpenCL is required.

In addition Autoscoper uses QT4, GLEW, and TIFFLIB. If the option AUTOBUILD_DEPENDENCIES is set GLEW and TIFFLIB will be automatically downloaded and installed.

You will also need Git and CMAKE

# Installation Instructions #

## WINDOWS ##

Prerequisites
- CUDA toolkit or OpenCL
- Cmake (https://cmake.org/)
- git (https://git-scm.com/downloads)
- QT 4.8 

Build
1. Clone the git repository (https://bitbucket.org/xromm/autoscoper-v2.git)
2. Run cmake and choose a source and the build folder for autoscoper and click configure. On Windows choose a 64bit build of the Visual Studio version you have installed. Dependencies to tiff and glew will be installed automatically and the other dependencies should be found automatically. By default it will use OpenCL if you want to use CUDA instead set BUILD_WITH_CUDA. 
3. click generate
4. click open project and build in Visual studio.
5. to install build the INSTALL project in VisualStudio. This will build autoscoper and installs it in your build folder in the subfolder install/bin/Debug or install/bin/RElease depending on which build was performed.

## LINUX ##

1. Install your GPGPU SDK. Either OpenCL or CUDA.
2. Install libtiff, glew, qt4 using your package manager
3. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate) 
4. Build using 'make'

## OS X ##

1. Install your GPGPU SDK. Either OpenCL or CUDA.
2. Install Macports and install qt4-mac, glew, tiff by using Macports. (Universal recommended)

```
sudo ports install glew +universal
sudo ports install qt4-mac +universal
sudo ports install tiff +universal
```
3. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate) 
4. Build with 'make'. (CMAKE_OSX_ARCHITECTURES recommended to set to x86_64;i386 for universal and CMAKE_OSX_DEPLOYMENT_TARGET to 10.6). Make sure the libs are set with the static versions (.a) and that LZMA is linked with the versions under /opt/local/lib/.