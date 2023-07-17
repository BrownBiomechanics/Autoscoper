# SlicerAutoscoper Pre-Processing Module

```{warning}
This module is currently under development, details are subject to change without notice.

The only supported tasks currently are the segmentation generation or loading and the partial volume cropping and extraction.
```

This tutorial covers the basic usage of the Pre-Processing module. The module is used to prepare the data for the Autoscoper module. The module is used to perform the following tasks:

* Segmentation generation or loading
* Partial volume cropping and extraction
* Virtual Radiograph (VRG) generation 
* Config file generation

This tutorial assumes that you have already loaded the data into Slicer. If you have not done so, please refer to the [Slicer Documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html) for more information.


## UI Overview

To get the to the pre-processing module, enter the AutoscoperM module located in the `Tracking` category. The pre-processing module is the second tab labeled `Autoscoper Pre-Processing`.

![Pre-Processing Module UI Overview](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_uiOverview.png)

## General Inputs

Currently, the pre-processing module requires two inputs:

* Volume Node: The volume node that contains the CT data
* Output Directory: The primary output directory for the pre-processing module, see the [](./custom-data.md#recommended-file-structure) for more information.

## Segmentations

You have three options for segmentations:

* Auto-generate: This option will automatically generate a segmentation based on the volume node. 
* Load: This option will allow you to select a directory containing segmentations. 
* Use Slicer to create new segmentation.
    * See the [Image Segmentation](https://slicer.readthedocs.io/en/latest/user_guide/image_segmentation.html) page of the Slicer documentation for more information.

### Auto Generated Segmentations

To auto-generate a set of segments for your CT data, select the `Automatic Segmentation` option and select a threshold value. The default value is 700 HU, but this value will vary depending on the CT data.

To pick a threshold value, use the data probe (located in the bottom left corner of the Slicer UI) move your mouse pointer along the volume. The data probe will display the HU value of the voxel under the mouse pointer. Determine the average HU value of the exterior surface of the bone and the average HU value of the flesh and muscle. The threshold value should be high enough to exclude the flesh and muscle, but low enough to include the exterior surface of the bone. Do not worry about the interior of the bone, the auto segmentation will fill in the interior of the bone.

![Data Probe](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_dataProbe.gif)

In the above video, we can see that the flesh and muscle have an average HU value of -500 to 300 HU. The exterior surface of the bone has an average HU value of 2000-3000 HU. The default value of 700 HU is a good starting point for this data.

```{warning}
Auto-generating the segmentations will take a long time. It is recommended that you use the `Load` option if you have already generated the segmentations.
```

Once you have selected a threshold value, click the `Generate Segmentations` button. This will generate a segmentation for each bone in the volume. The segmentations will be named `Segment_1`, `Segment_1_2`, etc.

![Auto Generated Segmentations](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_segmentationResults.png)

Once the segmentations have been generated, you can use the `Segment Editor` module to edit the segmentations. See the [Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html) page of the Slicer documentation for more information.

Switching to the `Data` module will allow you to view the segmentations. Here you can remove any segmentations that are not needed. You can also rename the segmentations to something more meaningful.

### Load Segmentations

To load a batch of select the `Batch Load from File` option, you can then select a directory below. The directory should contain a set of segmentations in the following format:

```
*.seg.nrrd
*.nrrd
*.wrl
*.iv
*.vtk
*.stl
```
Once the directory has been selected, click the `Generate Segmentations` button.

All of the segmentation files will be loading as individual segments in a single segmentation node. 

```{warning}
Segmentations loaded using this method may require a transformation to be applied for them to align with the volume node. See the [Transforms](https://slicer.readthedocs.io/en/latest/user_guide/modules/transforms.html) page of the Slicer documentation for more information.
```

## Partial Volume Cropping and Extraction

Once segmentations have been generated or loaded, you can use the `Partial Volume Generation` section to crop and extract the partial volumes. Simply, define the `Output Directory`, under the `General Inputs` section, select a `Segmentation Node`, and click the `Generate Partial Volumes` button.

This will generate an individual partial volume for each segment in the segmentation node. Partial volumes will be saved as a grayscale tif file, the filename will be the name of the segment. The partial volumes will be saved in the `Output Directory` in the following format:

```
Output Directory
|-- Volumes
|   |-- Segment_1.tif
|   |-- Segment_2.tif
|   |-- Segment_3.tif
|   |-- ...
|   |-- Segment_n.tif
```
