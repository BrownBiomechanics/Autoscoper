<!--- Autoscoper documentation master file, created by
   sphinx-quickstart on Mon Apr 10 14:29:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

# Welcome to Autoscoper's documentation!

## What is Autoscoper?

Autoscoper enables the 3D tracking of structures within multiple imaging modalities including single, bi-, and multi-plane videoradiography. Current development is underway which will expand tracking capabilities to sequential CT, 4DCT, and MRI volume sets.

Autoscoper has been used for tracking the shoulder, spine, wrist, hip, knee, and ankle joints.

## License

The Autoscoper software is distributed under a BSD-style open source license that is broadly compatible with the Open Source Definition by [The Open Source Initiative](https://opensource.org/) and contains no restrictions on legal uses of the software. For more information, see the [Autoscoper License File](https://github.com/BrownBiomechanics/Autoscoper/blob/main/LICENSE)

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
user-interface.md
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
