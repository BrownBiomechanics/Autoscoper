<!--- Autoscoper documentation master file, created by
   sphinx-quickstart on Mon Apr 10 14:29:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

# Welcome to Autoscoper's documentation!

## About

### What is Autoscoper?

Autoscoper enables the 3D tracking of structures within multiple imaging modalities including single, bi-, and multi-plane videoradiography. Current development is underway which will expand tracking capabilities to sequential CT, 4DCT, and MRI volume sets.

Autoscoper has been used for tracking the shoulder, spine, wrist, hip, knee, and ankle joints.

Autoscoper is distributed through SlicerAutoscoperM (SAM), a free, open source and multi-platform 3D Slicer extension. To learn more about the relationship between Autoscoper and AutoscoperM, see https://autoscoperm.slicer.org/.

### License

The Autoscoper software is distributed under a BSD-style open source license that is broadly compatible with the Open Source Definition by [The Open Source Initiative](https://opensource.org/) and contains no restrictions on legal uses of the software. For more information, see the [Autoscoper License File](https://github.com/BrownBiomechanics/Autoscoper/blob/main/LICENSE)

#### Historical Notes about the License

The Autoscoper license is an adaptation of the BSD-3 license with customized additions to the Disclaimer of Warranty and Limitation of Liability that were drafted by Brown University lawyers in 2011. These additional clauses go beyond the standard BSD-3 disclaimer by explicitly stating that Brown University provides no warranties or representations of any kind, including those for design, merchantability, and fitness for purpose, and disclaims any errors or infringement on third-party proprietary rights. The Autoscoper license also limits Brown University's liability for damages arising out of the use of the software, including direct and indirect damages, as well as any loss of data or profits.

### History

In November 2020, the development was transitioned from bitbucket to GitHub. See https://github.com/BrownBiomechanics/Autoscoper/

From 2018 to 2020, Dr.Bardiya Akhbari from Brown University maintained the software on bitbucket, worked on addressing the reported issues, further optimized the CUDA kernels, and implemented support for the Particle Swarm Optimization (PSO) algorithm to improve the accuracy and speed of registration.

From 2014 to 2018, Autoscoper v2 was developed at Brown University by Dr. Benjamin Knorlein. Multi-bone tracking and batch processing features were added, a socket-based protocol was designed for interaction with third-party software (mainly Matlab), the build system was transtioned to CMake and the UI was transitioned from Qt4 and GTK to Qt5. Version 2 combined the sources of both the CUDA and OpenCL versions and allows usage of either one. The source code was organized at `https://bitbucket.org/xromm/autoscoper-v2`.

From 2012 to 2013, Autoscoper v1 was developed at Brown University by Andy Loomis and Mark Howison. Andy developed the original CUDA version and Mark developed the OpenCL reimplementation. Source code was organized at `https://bitbucket.org/mhowison/xromm-autoscoper`.

## How to cite

When citing Autoscoper in your scientific research, please mention the following work to support increased visibility and dissemination of our software:

> Akhbari, B., Morton, A. M., Moore, D. C., Weiss, A-P. C., Wolfe, W. S., Crisco, J. J., 2019. Accuracy of Biplane Videoradiography for Quantifying Dynamic Wrist Kinematics, Journal of Biomechanics.
>
> See https://www.sciencedirect.com/science/article/abs/pii/S0021929019303847

For your convenience, you may use the following BibTex entry:

```bibtex
@article{AKHBARI2019120,
  title    = {Accuracy of biplane videoradiography for quantifying dynamic wrist kinematics},
  journal  = {Journal of Biomechanics},
  volume   = {92},
  pages    = {120-125},
  year     = {2019},
  issn     = {0021-9290},
  doi      = {https://doi.org/10.1016/j.jbiomech.2019.05.040},
  url      = {https://www.sciencedirect.com/science/article/pii/S0021929019303847},
  author   = {Bardiya Akhbari and Amy M. Morton and Douglas C. Moore and Arnold-Peter C. Weiss and Scott W. Wolfe and Joseph J. Crisco},
keywords = {Biplane videoradiography, Wrist kinematics, Accuracy study, Markerless tracking},
}
```

## Contact Us

It is recommended to post any questions, bug reports, or enhancement requests in the `SlicerAutoscoperM` category on the [Slicer forum](https://discourse.slicer.org/c/community/slicerautoscoperm/30).

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started.md
adv-topics/index.md
user-interface.md
tutorials/index.md
file-specifications/index.md
socket-control-libraries/index.md
developer-guide/index.md
```

## Sample Data

Sample data is available for download from the [SlicerAutoscoperM Sample Data](tutorials/sample-data.md#downloading-sample-data) page. Currently available sample data includes:

* Wrist BVR data - This was part of the data used in the [Akhbari et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0021929019303847) paper.
  * Three frames of movement are included in the sample data.
  * Four DRRs are included in the sample data. The radius, ulna, third meta-carpal, and a combined second and third meta-carpal are included.
* Knee BVR data - This data was provided by Jill Beveridge.
  * Three frames of movement are included in the sample data.
  * Two DRRs are included in the sample data. The femur and tibia are included.
* Ankle BVR data - This data was provided by Michael Rainbow.
  * Three frames of movement are included in the sample data.
  * Three DRRs are included in the sample data. The tibia, talus, and calcaneus are included.

# Indices and tables

* {ref}`genindex`
* {ref}`search`
