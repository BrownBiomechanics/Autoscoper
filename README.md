# Autoscoper

Autoscoper enables the 3D tracking of structures within multiple imaging modalities including single, bi-, and multi-plane videoradiography, sequential CT volume images, 4DCT image volumes sets, and MRI volume sets.

Autoscoper has been used for tracking the shoulder, spine, wrist, hip, knee, and ankle joints.

## Installation

Autoscoper is distributed through SlicerAutoscoperM (SAM), a free, open source and multi-platform 3D Slicer extension. To learn more about the relationship between Autoscoper and AutoscoperM, see https://autoscoperm.slicer.org/.

## Documentation

Autoscoper provides documentation to help users get started and use the program effectively. The documentation is hosted on [Read the Docs](https://autoscoper.readthedocs.io). For convenience, direct links are also provided below.

* [Getting Started][getting-started]
* [Tutorials][tutorials]

[getting-started]: https://autoscoper.readthedocs.io/en/latest/getting-started.html
[tutorials]: https://autoscoper.readthedocs.io/en/latest/tutorials/index.html

The documentation includes step-by-step tutorials, explanations of important concepts, and detailed information on how to use the program.

## License

The Autoscoper software is distributed under a BSD-style open source license that is broadly compatible with the Open Source Definition by [The Open Source Initiative](https://opensource.org/) and contains no restrictions on legal uses of the software.

### Historical notes about the license

The Autoscoper license is an adaptation of the BSD-3 license with customized additions to the Disclaimer of Warranty and Limitation of Liability that were drafted by Brown University lawyers in 2011. These additional clauses go beyond the standard BSD-3 disclaimer by explicitly stating that Brown University provides no warranties or representations of any kind, including those for design, merchantability, and fitness for purpose, and disclaims any errors or infringement on third-party proprietary rights. The Autoscoper license also limits Brown University's liability for damages arising out of the use of the software, including direct and indirect damages, as well as any loss of data or profits.

## History

In November 2020, the development was transitioned from bitbucket to GitHub. See https://github.com/BrownBiomechanics/Autoscoper/

From 2018 to 2020, [Dr. Bardiya Akhbari][bardiya] from Brown University maintained the software on bitbucket, worked on addressing the reported issues, further optimized the CUDA kernels, and implemented support for the Particle Swarm Optimization (PSO) algorithm to improve the accuracy and speed of registration.

From 2014 to 2018, Autoscoper v2 was developed at Brown University by [Dr. Benjamin Knorlein][knorlein]. Multi-bone tracking and batch processing features were added, a socket-based protocol was designed for interaction with third-party software (mainly Matlab), the build system was transitioned to CMake and the UI was transitioned from Qt4 and GTK to Qt5. Version 2 combined the sources of both the CUDA and OpenCL versions and allows usage of either one. The source code was organized at `https://bitbucket.org/xromm/autoscoper-v2`.

From 2012 to 2013, Autoscoper v1 was developed at Brown University by Andy Loomis and Mark Howison. Andy developed the original CUDA version and Mark developed the OpenCL reimplementation. Source code was organized at `https://bitbucket.org/mhowison/xromm-autoscoper`.

[bardiya]: https://www.researchgate.net/profile/Bardiya_Akhbari
[knorlein]: https://www.ccv.brown.edu/about/staff
