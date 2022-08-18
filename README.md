# Autoscoper

Autoscoper enables the 3D tracking of structures within multiple imaging modalities including single, bi-, and multi-plane videoradiography, sequential CT volume images, 4DCT image volumes sets, and MRI volume sets.

Autoscoper has been used for tracking the shoulder, spine, wrist, hip, knee, and ankle joints.

## Installation

The installer for Autoscoper 2.7.1 are available [on the SimTk website](https://simtk.org/projects/autoscoper).

You need to install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads?), and update your graphics card driver to run the application.

## History

In November 2020, the development was transitioned from bitbucket to GitHub. See https://github.com/BrownBiomechanics/Autoscoper/

From 2018 to 2020, [Dr. Bardiya Akhbari][bardiya] from Brown University maintained the software on bitbucket, worked on addressing the reported issues, further optimized the CUDA kernels, and implemented support for the Particle Swarm Optimization (PSO) algorithm to improve the accuracy and speed of registration.

From 2014 to 2018, Autoscoper v2 was developed at Brown University by [Dr. Benjamin Knorlein][knorlein]. Multi-bone tracking and batch processing features were added, a socket-based protocol was designed for interaction with third-party software (mainly Matlab), the build system was transtioned to CMake and the UI was transitioned from Qt4 and GTK to Qt5. Version 2 combined the sources of both the CUDA and OpenCL versions and allows usage of either one. The source code was organized at `https://bitbucket.org/xromm/autoscoper-v2`.

From 2012 to 2013, Autoscoper v1 was developed at Brown University by Andy Loomis and Mark Howison. Andy developed the original CUDA version and Mark developed the OpenCL reimplementation. Source code was organized at `https://bitbucket.org/mhowison/xromm-autoscoper`.

[bardiya]: https://www.researchgate.net/profile/Bardiya_Akhbari
[knorlein]: https://www.ccv.brown.edu/about/staff
