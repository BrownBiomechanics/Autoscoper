# Compiling Instructions

## WINDOWS

Prerequisites

- [CUDA toolkit v10.1 or later](https://developer.nvidia.com/cuda-downloads?)
- [CMake](https://cmake.org/)
- [Git](https://git-scm.com/downloads)
- [Qt5](https://www.qt.io/download-open-source): Download Qt universal installer and install Qt 5.15.2 components.
- Update your graphics card driver.

### GUI-Build

1. Clone the [GitHub repository](https://github.com/BrownBiomechanics/Autoscoper).
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

### Cli-Build (Using Powershell)

1. Clone the [GitHub repository](https://github.com/BrownBiomechanics/Autoscoper).
2. Enter the repos directory
3. Make and enter a build directory `mkdir build` and `cd build`
4. Configure project `cmake .. -DAutoscoper_SUPERBUILD=ON -DAutoscoper_RENDERING_BACKEND="CUDA"/"OpenCL" -DQt5_DIR="path/to/Qt5Config.cmake"`
5. Build external dependencies `cmake --build . --config Release/Debug`

Optionally Install the project:

6. Install project `cmake --build Autoscoper-build --target install --config Release/Debug`
7. The autoscoper.exe will be in the folder build/install/bin/Debug or build/install/bin/Release depending on which build was performed.

## GNU/Linux systems

1. Clone the [GitHub repository](https://github.com/BrownBiomechanics/Autoscoper).
2. Create a build folder in the autoscoper folder and run 'ccmake ../.' from the build folder (configure and generate)
3. Build using 'make' in the build folder.

NOTE for HPC SERVERs: You need to use VNC or another application that gives you a display access. Autoscoper will not run if your HPC server does not have display and GPU access.

## MAC OS - CUDA Only

_These instructions are untested and maybe obsolete._

1. Clone the [GitHub repository](https://github.com/BrownBiomechanics/Autoscoper).
2. Create a build folder in the autoscoper folder, open CMake and use XCode as compiler.
3. When receive an error, modify the fields:
  1. CMAKE_OSX_ARCHITECTURES recommended to set to x86_64
  2. CMAKE_OSX_DEPLOYMENT_TARGET to 10.15 (or your mac_os version)
  3. If received an error for Qt5_DIR, search for (Qt5Config.cmake) on your hard drive and write its location in the field.
4. After generating the configured file, open XCode and compile the application

## Docker Image

1. Clone the [GitHub repository](https://github.com/BrownBiomechanics/Autoscoper)
2. Install [Docker](https://www.docker.com/products/docker-desktop)
3. If running Windows Subsystem for Linux (WSL), install [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/) for GUI passthrough.
4. Open a terminal and navigate to the docker subfolder of the repository
5. Run `docker build -t "autoscoper_dev_ubuntu:latest" -f ./UbuntuDockerFile .`
6. Find your IP address (using `ipconfig` on Windows or `ifconfig` on Unix)
7. Run `docker run --rm -it --gpus all -e DISPLAY=<IP>:0.0 --name autoscoper_ubuntu autoscoper_dev_ubuntu:latest`
