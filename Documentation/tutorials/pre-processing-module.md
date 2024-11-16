# Pre-Processing Module

```{warning}
This module is currently under development, and details are subject to change without notice.

Currently, the module supports the following tasks:
- Segmentation generation or loading
- Partial volume cropping and extraction

In the future, we will add support for the following tasks:
- Config file generation
```

## Introduction

The Pre-Processing module in Autoscoper is a crucial step in preparing data for further analysis within the Autoscoper module. In this tutorial, we'll explore the basic usage of the Pre-Processing module, detailing its functionalities and how to use them effectively.

Before diving into pre-processing, ensure that you have already loaded the data into 3D Slicer. If you haven't done this yet, you can find detailed instructions in the [Slicer Documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html).

## Accessing the Pre-Processing Module

To access the Pre-Processing module, open the AutoscoperM module located in the `Tracking` category. Next, navigate to the second tab labeled `Autoscoper Pre-Processing`.

![Pre-Processing Module UI Overview](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_overview.png)

## General Inputs

The Pre-Processing module requires two inputs:

* Volume Node: This refers to the volume node containing the CT data.
* Output Directory: The primary output directory for the pre-processing results. For more information on the recommended file structure, see the [](./custom-data.md#recommended-file-structure) section.

## Segmentations

You have three options for segmentations:

* Auto-generate: Automatically generates segmentations based on the volume node.
* Load: Allows you to load existing segmentations from a directory.
* Use Slicer to create new segmentation: Utilizes the built-in Slicer tools for segmentation. For more information see the [Image Segmentation](https://slicer.readthedocs.io/en/latest/user_guide/image_segmentation.html) page of the Slicer documentation.

### Auto-Generated Segmentations

To auto-generate a set of segments for your CT data, select the `Automatic Segmentation` option and specify a threshold value. The default value is 700 Hounsfield Units (HU), but this value may vary depending on your CT data.

To determine an appropriate threshold value, use the data probe (located in the bottom left corner of the Slicer UI) to explore the volume. The data probe displays the HU value of the voxel under the mouse pointer. Determine the average HU value of the exterior surface of the bone and the average HU value of the flesh and muscle. The threshold value should be high enough to exclude the flesh and muscle, but low enough to include the exterior surface of the bone. Do not worry about the interior of the bone, the auto segmentation will fill in the interior of the bone.

![Data Probe](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_dataProbe.gif)

In the example above, we can see that the flesh and muscle have an average HU value of -500 to 300 HU. The exterior surface of the bone has an average HU value of 2000-3000 HU. The default value of 700 HU is a suitable starting point for this data.

```{warning}
Auto-generating the segmentations will take a long time. It is recommended that you use the `Load` option if you have already generated the segmentations.
```

Once you have selected a threshold value, click the `Generate Segmentations` button. This will generate a segmentation for each bone in the volume and they will be named `Segment_1`, `Segment_1_2`, etc.

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

## Partial Volume Cropping and Extraction

Once you have generated or loaded the segmentations, you can use the `Partial Volume Generation` section to crop and extract partial volumes. Simply, define the `Output Directory`, select a `Segmentation Node` under the `General Inputs` section, and click the `Generate Partial Volumes` button.

This will generate an individual partial volume for each segment in the segmentation node. The partial volumes will be saved as a grayscale TIF files, with filenames corresponding to the names of the segments. The partial volumes will be stored in the `Output Directory` as follows:

```
Output Directory
|-- Volumes
|   |-- Segment_1.tif
|   |-- Segment_2.tif
|   |-- Segment_3.tif
|   |-- ...
|   |-- Segment_n.tif
```
