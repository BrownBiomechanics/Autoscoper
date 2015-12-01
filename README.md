This is a redesign by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff) of the autoscoper developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper),

The new Version combines both the CUDA and OpenCL version and allows usage of either of the 2 Frameworks. 

Furthermore in order to simplify installation and compilation the new Version uses CMake while the UI has been changed to QT4.

Installation Instructions

APPLE
1. Install Macports

2. Install qt4-mac, glew, tiff by using Macports. (Universal recommended)
sudo ports install glew +universal
sudo ports install qt4-mac +universal
sudo ports install tiff +universal

3a. Download OpenCV and create a build folder.
3b. Run ccmake ../. from the build folder. Set Shared libs to false  (configure and generate) and build with make. (CMAKE_OSX_ARCHITECTURES recommended to set to x86_64;i386 for universal and CMAKE_OSX_DEPLOYMENT_TARGET to 10.6).
3c. Copy the include folder to your build folder.

4a. Create a build folder in the autoscoper folder.
4b. Run ccmake ../. from the build folder (configure and generate) and build with make. (CMAKE_OSX_ARCHITECTURES recommended to set to x86_64;i386 for universal and CMAKE_OSX_DEPLOYMENT_TARGET to 10.6). Make sure the libs are set with the static versions (.a) and that LZMA is linked with the versions under /opt/local/lib/.

WINDOWS

1. Download libtiff, glew, qt4, opencv and compile if necessary

2. Run cmake and configure

3. Build using Visual studio


LINUX 


1. Download libtiff, glew, qt4, opencv and compile if necessary

2. Run cmake and configure

3. Build using make