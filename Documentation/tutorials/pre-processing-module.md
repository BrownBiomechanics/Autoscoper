# Pre-Processing Module

```{warning}
The Pre-Processing module is still under development, details are subject to change without notice.
```

## Introduction

The Pre-Processing module in Autoscoper is a crucial step in preparing data for further analysis within the Autoscoper module. In this tutorial, we'll explore the basic usage of the Pre-Processing module, detailing its functionalities and how to use them effectively.

Before diving into pre-processing, ensure that you have already loaded the data into 3D Slicer. If you haven't done this yet, you can find detailed instructions in the [Slicer Documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html).

## Accessing the Pre-Processing Module and UI Overview

To access the Pre-Processing module, open the AutoscoperM module located in the `Tracking` category. Next, navigate to the second tab labeled `Autoscoper Pre-Processing`.

The UI is broken into six sections:
1. General Inputs
2. Segmentation Generation
3. VRG Generation - Manual Camera Placement
4. VRG Generation - Automatic Camera Placement
5. Partial Volume Generation
6. Config file Generation.

![UI Overview](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_overview.png)


## General Inputs

The Pre-Processing module requires three inputs:

* Volume Node: This refers to the volume node containing the CT data.
* Output Directory: The primary output directory for the pre-processing results. For more information on the recommended file structure, see the [](./custom-data.md#recommended-file-structure) section.
* Trial Name: The name of the trial. This will be used to name the output files.

There are also a wide array of optional inputs that can be used to customize the various sub-directories and file parameters. You can find the optional inputs under the `Advanced Options` section of the `General Inputs` tab.

![General Inputs](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_generalInput.png)

## Segmentations

You have three options for segmentations:

* Auto-generate: Automatically generates segmentations based on the volume node.
* Load: Allows you to load existing segmentations from a directory.
* Use Slicer to create new segmentation: Utilizes the built-in Slicer tools for segmentation. For more information see the [Image Segmentation](https://slicer.readthedocs.io/en/latest/user_guide/image_segmentation.html) page of the Slicer documentation.

![Segmentations](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_segmentation.png)

### Auto-Generated Segmentations

To auto-generate a set of segments for your CT data, select the `Automatic Segmentation` option and specify a threshold value and a margin size. The default value is 700 Hounsfield Units (HU), but this value may vary depending on your CT data.

To determine an appropriate threshold value, use the data probe (located in the bottom left corner of the Slicer UI) to explore the volume. The data probe displays the HU value of the voxel under the mouse pointer. Determine the average HU value of the exterior surface of the bone and the average HU value of the flesh and muscle. The threshold value should be high enough to exclude the flesh and muscle, but low enough to include the exterior surface of the bone. Do not worry about the interior of the bone, the auto segmentation will fill in the interior of the bone.

![Data Probe](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_dataProbe.gif)

In the example above, we can see that the flesh and muscle have an average HU value of -500 to 300 HU. The exterior surface of the bone has an average HU value of 2000-3000 HU. The default value of 700 HU is a suitable starting point for this data.

To pick a margin size, find the largest bone in the volume and measure its diameter. The margin size should be roughly a tenth of the diameter of the bone. For example, if the largest bone has a diameter of 100 mm, the margin size should be 10 mm. It is recommended to try a few different margin sizes to see which one works best for your data. See the [measurement section](https://slicer.readthedocs.io/en/latest/user_guide/modules/markups.html#measurements-section) of the Slicer documentation for more information on measuring distances.

```{warning}
Auto-generating the segmentations will take a long time. It is recommended that you use the `Load` option if you have already generated the segmentations.
```

Once you have selected a threshold value and a margin size, click the `Generate Segmentations` button. This will generate a segmentation for each bone in the volume and they will be named `Segment_1`, `Segment_1_2`, etc.

![Auto Generated Segmentations](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_segmentationResults.png)

After generating the segmentations, you can further edit them using the `Segment Editor` module. For more information on segmentation editing, refer to the [Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html) page in the Slicer documentation.

Switching to the `Data` module allows you to view and manage the segmentations. Here you can remove any unnecessary segmentations and rename them for clarity.

### Load Segmentations

To load existing segmentations, select the `Batch Load from File` option and specify the directory containing the segmentations. The directory should contain segmentation files in one of the following formats:

```
*.seg.nrrd
*.nrrd
*.wrl
*.iv
*.vtk
*.stl
```

After selecting the directory, click the `Generate Segmentations` button.

The loaded segmentations will appear as individual segments within a single segmentation node.

```{warning}
Segmentations loaded using this method may require a transformation to align them with the volume node. For more information on applying transformations, refer to the [Transforms](https://slicer.readthedocs.io/en/latest/user_guide/modules/transforms.html) page in the Slicer documentation.
```

## Virtual Radiograph Generation

The Virtual Radiograph Generation (VRG) section allows you to generate virtual radiographs from the CT data. There are two methods for generating VRGs: manual camera placement and automatic camera placement.

![Virtual Radiograph Generation](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_vrg.png)

### Manual Camera Placement

The manual camera placement method allows you to manually place cameras around the volume and generate VRGs from the camera positions. There are four inputs for this method:

* Segmentation Node: The segmentation node containing the segments to be used for the VRGs. This should be a complete segmentation of all the bones in the volume.
* Camera Positions: A points list node  containing the camera positions. This can be generated using the `Markups` module. For more information on the `Markups` module, refer to the [Markups](https://slicer.readthedocs.io/en/latest/user_guide/modules/markups.html) page in the Slicer documentation.
* View Angle: The angle of the camera frustum. The default value is 30 degrees.
* Clipping Range: The clipping range of the camera frustum. The default value is (0.1, 300.0).

To generate the VRGs, input the required parameters and click the `Generate VRGs from Markups` button. This will generate a VRG for each camera position and save them in the `VRG` sub-directory of the `Output Directory`. This will also generate the corresponding camera parameters file and save it in the `Camera` sub-directory of the `Output Directory`.

### Automatic Camera Placement

The automatic camera placement method will find the optimal camera positions for generating VRGs. There are four inputs for this method:

* Number of optimized cameras: The number of cameras to be used for the VRGs. The default value is 2.
* Number of initial cameras: The number of initial cameras to be used for the optimization. The default value is 50.
* Segmentation Node: The segmentation node containing the segments to be used for the VRGs. This should be a complete segmentation of all the bones in the volume.
* Camera offset: The offset of the camera from the center of the volume. The default value is 400 mm.

To generate the VRGs, input the required parameters and click the `Generate VRGs` button. This will generate a VRG for each of the possible camera positions and save them in the `VRG-Temp` sub-directory. It will then evaluate the VRGs and select the optimal camera positions. The optimal VRGs will be saved in the `VRG` sub-directory of the `Output Directory`. This will also generate the corresponding camera parameters file and save it in the `Camera` sub-directory of the `Output Directory`.

```{note}
The `VRG-Temp` sub-directory will be deleted after the optimization is complete. If you wish to keep the VRGs generated during this step, de-select the `Delete temporary VRGs` option in the `Advanced Options` section of the `General Inputs` tab.
```


## Partial Volume Cropping and Extraction

![Partial Volume Cropping and Extraction](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_pv.png)

Once you have generated or loaded the segmentations, you can use the `Partial Volume Generation` section to crop and extract partial volumes. Simply, define the `Output Directory`, select a `Segmentation Node` under the `General Inputs` section, and click the `Generate Partial Volumes` button.

This will generate an individual partial volume for each segment in the segmentation node. The partial volumes will be saved as a grayscale TIF files, with filenames corresponding to the names of the segments. The partial volumes will be saved in the `Volumes` sub-directory of the `Output Directory`. This will also generate the corresponding transformation files and save them in the `Transforms` sub-directory of the `Output Directory`. This transformation file can be used to align the partial volumes with the original volume node.

To load in previously generated partial volumes, ensure the correct `Output Directory` under the `General Inputs` section is selected and click the `Load Partial Volumes` button. This will load in the
partial volumes and their corresponding transformations.


## Config File Generation

![Config File Generation](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_config.png)

The `Config File Generation` section allows you to generate the config file required for the Autoscoper module. There are two optional inputs for this section:

* Optimization offsets: The offsets to be used for the optimization. The default value is 0.1 for each degree of freedom.
* Volume Flip: The axis to flip the volume along. The default value is False for each axis.

Ensuring that the `Output Directory`, under the `General Inputs` section, contains a `Camera`, `Volumes`, and `VRG` sub-directory, click the `Generate Config File` button. This will generate the config file and save it in the `Output Directory`, named `{trial_name}.cfg`. This will automatically place the path to the config file in the `Config File` input box on the `Autoscoper Control` tab of the `AutoscoperM` module.
