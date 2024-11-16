# Getting Started

This page contains information on how to obtain and install AutoscoperM, as well as how to get started using the software.

## System Requirements

AutoscoperM is a module for [3D Slicer](https://download.slicer.org/), a free, open-source software platform for medical image computing. For more information on the system requirements for 3D Slicer, please see the [3D Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#system-requirements).

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
  * Four partial volumes are included in the sample data. The radius, ulna, third meta-carpal, and a combined second and third meta-carpal are included.
* Knee BVR data - This data was provided by Jill Beveridge.
  * Three frames of movement are included in the sample data.
  * Two partial volumes are included in the sample data. The femur and tibia are included.
* Ankle BVR data - This data was provided by Michael Rainbow.
  * Three frames of movement are included in the sample data.
  * Three partial volumes are included in the sample data. The tibia, talus, and calcaneus are included.

## Glossary

This section defines terms that are commonly used by Autoscoper and SlicerAutoscoperM

* **3D Computed Tomography (CT)**: A computerized X-ray imaging method that produces volumetric density data of a region of interest. The data is typically saved in DICOM format.
* **4D CT**: A temporal sequence 3D CT scans, capturing both spatial and temporal changes.
* **Biplaner Videoradiography (BVR)**: A system utilizing two calibrated X-ray sources and high-speed video cameras to simultaneously capture radiographs from two perspectives. The camera models derived from the calibrations enable the construction of a unified 3D World construction from the 2D perspective views.
* **Rigid Body**: In the context of 3D imaging and motion tracking, a Rigid Body refers to a solid object, such as a bone or implant, that does not deform under normal conditions. Within a 3DCT (three-dimensional computed tomography) scan, the rigid body is the specific region of interest that users identify and isolate. This segmented data is often exported as a bone mesh model in STL (stereolithography) format. The primary purpose of Autoscoper is to track the kinematics (both rotation and translation) of the rigid body as it moves through three-dimensional space.
* **Partial Volume (PVOL)**: A volumetric image data file saved as a TIFF stack. The PVOL represents a subset of 3D CT scan data, defined by a segmented rigid body and cropped to its dimensions.
* **AUT Model (AUT)**: A representation of the rigid body STL file, as parsed and visualized within the Autoscoper 3D World.
* **Digitally Reconstructed Radiograph (DRR)**: A synthetic radiograph generated when a PVOL is loaded into Autoscoper. Perspective camera models cast the volumetric density data onto the same image plane as the radiographs, simulating the radiograph digitally.
* **Radiograph**: A 2D image produced by X-rays.
* **Sequential 3D CT**: A series of 3D CT scans acquired from the same region of interest (such as a joint or limb) in different positions. Using the Sequences module, these independent volumes can be visualized and processed together as a series.
* **TFM**: An ITK transformation file with the extension `.tfm`.
* **TRA**: A transform file with the extension `.tra` exported by Autoscoper. It contains a Nx16 transformation matrix of rigid body kinematics. Each one of the N lines represents a frame of BVR data optimized by Autoscoper. The TRA transforms can be applied to AUT models to visualized 3D motion of the rigid body and calculate motion.
