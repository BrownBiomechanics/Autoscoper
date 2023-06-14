# Getting Started

This page contains information on how to obtain and install AutoscoperM, as well as how to get started using the software.

## System Requirements

AutoscoperM is a module for [3D Slicer](https://download.slicer.org/), a free, open-source software platform for medical image computing. For more information on the system requirements for 3D Slicer, please see the [3D Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#system-requirements).

```{note}
AutoscoperM currently only supports Windows and Linux operating systems. MacOS support is planned for the near future.
```

### OpenCL Requirements

A graphics card that supports OpenCL 1.2 or higher is required to run AutoscoperM. For more information on OpenCL, please see the [OpenCL documentation](https://www.khronos.org/opencl/). Please check your graphics card's specifications to ensure that it supports OpenCL 1.2 or higher. If your graphics card supports OpenCL 1.2 or higher, please make sure to update to the latest version of your graphics card drivers.


## Installing AutoscoperM

1. Download and install the latest **preview** release of [3D Slicer](https://download.slicer.org/). For more information on installing 3D Slicer, please see the [3D Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer).
2. Once installed, open 3D Slicer and go to the [Extensions Manager](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html).
3. Search for and install the `AutoscoperM` extension.
4. Restart 3D Slicer. The AutoscoperM module will be available in the application, under the `Tracking` category.

```{hint}
If you are using AutoscoperM on a remote computer, you will likely need to setup a [remote GPU](./adv-topics/remote-gpu-setup.md) to enable GPU acceleration.
```

## Tutorials

A series of tutorials are available to help you get started using AutoscoperM. These tutorials are available on the [AutoscoperM Tutorials](./tutorials/index.md) page.

## Sample Data

Sample data is available for download from the [SlicerAutoscoperM Sample Data](/tutorials/loading-and-tracking.md#downloading-sample-data) page. Currently available sample data includes:

* Wrist BVR data - This was part of the data used in the [Akhbari et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0021929019303847) paper. 
  * Three frames of movement are included in the sample data.
  * Four DRRs are included in the sample data. The radius, ulna, third meta-carpal, and a combined second and third meta-carpal are included.
* Knee BVR data - This data was provided by Jill Beveridge.
  * Three frames of movement are included in the sample data.
  * Two DRRs are included in the sample data. The femur and tibia are included.
* Ankle BVR data - This data was provided by Michael Rainbow.
  * Three frames of movement are included in the sample data.
  * Three DRRs are included in the sample data. The tibia, talus, and calcaneus are included.
