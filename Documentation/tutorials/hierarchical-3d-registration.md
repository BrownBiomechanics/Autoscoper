# Hierarchical 3D Registration (3DH) Module

```{warning}
This module is currently under development, and details are subject to change without notice.

Currently, the module supports the following tasks:
- Registration of a 3D CT image to a sequence of 3D CT data
- Registration of a 3D CT image to 4D CT data

In the future, we will add support for the following tasks:
- Automatic import and export of registration results
- Rerunning or resuming interrupted registration runs
```

## Introduction

The Hierarchical 3D Registration module in Autoscoper is designed for image-based skeletal and implant motion tracking that computes six degree-of-freedom kinematics from static and dynamic computed tomography (3DCT and 4DCT, respectively) image datasets. The module automates the registration process, where the position and orientation of segmented bones across serial volume images of the same subject are determined.

The Hierarchical 3D Registration module utilizes a robust image registration algorithm based on rigid transformations. The algorithm is inspired by the method proposed by [Marai et al. 2006](http://dx.doi.org/10.1109/TMI.2005.862151) and implemented using [ITKElastix](https://elastix.dev/index.php). The bone hierarchy defines the structure of objects to be registered, arranged to reflect the anatomical relationships between parent and child bones. The process starts by defining regions of interest (ROIs) around the bones in the static 3D CT image (Source Volume), which will be compared against the corresponding regions in each frame of the dynamic dataset (CT Sequence). The entire hierarchy is registered in each frame, traversed in a breadth-first manner  starting with the root bone. During the registration of a given bone from the source volume to a specific frame in the sequence, the module provides a initial transformation that serves as the starting point for the optimizer. Once the optimal transformation from the source region to the target region is found, it is propagated to all child bones in the hierarchy, adjusting the starting position for each subsequent bone in the current frame. This hierarchical approach enhances the optimization process by accounting for motion constraints and thereby improving the alignment accuracy throughout the sequence.

## Accessing the Hierarchical 3D Registration Module

3DSLicer version 5.8.0 or above is required for the Hierarchical 3D Registration Module.

To access the module, install the `Hierarchical 3D Registration` extension, following similar steps to the [AutoscoperM extension installation instructions](getting-started.md#installing-autoscoperm), then open the `Hierarchical 3D Registration` module located under the `Tracking` category in 3D Slicer.

## General Inputs and Outputs

The Hierarchical 3D Registration module requires four inputs:

* CT Sequence: A sequence node containing the serial CT volume images to be tracked
* Source Volume: A static 3D CT image to register from, the scalar volume node from which the model hierarchy is generated
* Model Hierarchy: The root node of the model hierarchy, representing the rigid objects to be registered
* Frames: The range of frames to be tracked

The output of the module is a sequence of transforms for each bone, mapping from the bone's pose in the source volume to its pose in each frame. The module also generates additional nodes in the scene used during the registration process, such as:
* Region of interest (ROI) nodes, used to define the regions to compare from the source volume to each sequence frame
* Cropped volumes based on the ROIs of the source volume and sequence frames

<!-- ![Hierarchical 3D Registration Module UI Overview](TODO.png) -->

### Preparing the Model Hierarchy

1) Load in the STL Models previously segmented from the Source Volume. If not yet processed, see the [Pre-Processing Auto-Generated Segmentation](tutorials/pre-processing-module.md#auto-generated-segmentations) SAM module.
2) Once loaded into the Scene, navigate to the Data module. Child nodes (Models) can be nested (drag and drop in Data module) under the Root node in accordance with the desired transform propagation.

![Creating a Model Hierarchy in the Data Module](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/model_hier_demo.gif)


### Preparing the Input Sequence Volume

Slicer automatically loads 4D CT data as sequences. Multiple static 3D CT Scans can be combined to form a Sequence. Load all scalar volume data into the scene, then [construct the sequence using the Sequence Module](https://slicer.readthedocs.io/en/latest/user_guide/modules/sequences.html#creating-sequences-from-a-set-of-nodes). Using the Sequences module under the Edit tab, Create a new Sequence and add desired Volume Data nodes from the available list.


### Registering

Once all input data is prepared and the desired registration parameters are configured, click the "Initialize Registration" button.

The initialization step creates the necessary ROI nodes for each bone in the hierarchy, and crops the relevant regions from the source volume for comparison with the target sequence frames. For this reason, proper alignment of the model hierarchy in the source volume is crucial for accurate registration.

Following initialization, you will be prompted to adjust the bone transform. A transform interaction widget will appear in the slice views and 3D view, allowing you to manually adjust the position and orientation of the model closer to its target position in the current frame. This will serve as the initial guess for the optimal transform between the volumes.

![Adjusting the Initial Transform Guess for Registration](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/Registering.gif)

If the change in position across frames is minimal enough to not require manual intervention, you can choose to skip it by selecting the "Skip Initial Guess Adjustment" option. This enables the registration process to proceed fully automatically. Use this option with caution, since the registration accuracy depends on the quality of the initial alignment.

Once the initial guess is set, click the "Set Initial Guess And Register" button to trigger the optimization process. As mentioned above, the module uses the adjusted transformation to register the source region of interest to its corresponding region in the target frame. The Elastix optimizer calculates the transform that best minimizes the distance between the two images. The result transformation is then propagated through hierarchy, improving the initial positions of all child nodes for the subsequent registration steps.

The registration workflow provides visual feedback through the Slicer scene, allowing you to interact and adjust the input transformations when necessary. You can monitor the process using the progress bar at the top of the menu and the help string displayed below while registration is ongoing.

![3D CT Registration Results](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/final_registration_success.png)

Upon completion, the sequences of transforms will contain the registration results. These transforms will be applied to the model nodes, so that all tracked objects are visually aligned across the sequence frames. You can navigate through the sequence using the Sequence Browser to view the registration results for different frames.


<!-- ![4D CT Registration Results](TODO.gif) -->
