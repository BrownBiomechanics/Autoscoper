This is a redesign by [Dr. Ben Knorlein](https://www.ccv.brown.edu/about/staff) of the autoscoper developed by Andy Loomis (original CUDA version) and [Mark Howison (OpenCL reimplementation)](https://bitbucket.org/mhowison/xromm-autoscoper),

The new Version combines both the CUDA and OpenCL version and allows usage of either of the 2 Frameworks. 

Furthermore in order to simplify installation and compilation the new Version uses CMake while the UI has been changed to QT4.

Installation Instructions

APPLE

1. Install Macports and install qt4-mac, glew, tiff by using Macports. (Universal recommended)
sudo ports install glew +universal
sudo ports install qt4-mac +universal
sudo ports install tiff +universal

2. Create a build folder in the autoscoper folder andrun ccmake ../. from the build folder (configure and generate) 

3. Build with make. (CMAKE_OSX_ARCHITECTURES recommended to set to x86_64;i386 for universal and CMAKE_OSX_DEPLOYMENT_TARGET to 10.6). Make sure the libs are set with the static versions (.a) and that LZMA is linked with the versions under /opt/local/lib/.

WINDOWS

1. Download libtiff, glew, glut, qt4 and compile if necessary

2. Create a build folder in the autoscoper folder. Run cmake-gui, Set source folder and build folder. Then configure and generate.

3. Build using Visual studio project from build folder


LINUX 


1. Install libtiff, glew, qt4, glut using your package manager

2. Create a build folder in the autoscoper folder and run ccmake ../. from the build folder (configure and generate) 

3. Build using make