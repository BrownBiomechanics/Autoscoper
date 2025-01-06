# Pre-Processing Module

```{warning}
This module is currently under development, and details are subject to change without notice.

Currently, the module supports the following tasks:
- Segmentation generation or loading
- Partial volume cropping and extraction
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
* Load: Allows you to load segmentations from existing STL files in a directory.
* Use Slicer to create new segmentation: Utilizes the built-in Slicer tools for segmentation. For more information see the [Image Segmentation](https://slicer.readthedocs.io/en/latest/user_guide/image_segmentation.html) page of the Slicer documentation.

### Auto-Generated Segmentations

To auto-generate a set of segments for your CT data, select the `Automatic Segmentation` option and specify a threshold value. The default value is 700 Hounsfield Units (HU), but a different value may be more ideal, depending on your CT data.

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

To load existing segmentations, select the `Segmentation from Model` option and specify the directory containing the segmentations. The directory should contain segmentation files in one of the following formats:

```
*.seg.nrrd
*.nrrd
*.wrl
*.iv
*.vtk
*.stl
```

After selecting the directory, click the `Import Models` button.

The loaded segmentations will appear in Slicer as individual segments within a single segmentation node.

```{warning}
Segmentations loaded using this method may require a transformation to align them with the volume node. For more information on applying transformations, refer to the [Transforms](https://slicer.readthedocs.io/en/latest/user_guide/modules/transforms.html) page in the Slicer documentation.
```

```{tip}
Cropping the input volume to a smaller region of interest (ROI) can help reduce the computation time and the amount of irrelevant output generated. For more details on volume cropping, see the [Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/modules/cropvolume.html)
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

```{warning}
Partial Volumes generated using this method must be loaded into Slicer via the `Load Partial Volumes` button, rather than through the `Add Data` widget or dragging and dropping the files. This is because the generated TIFF files do not inherently store spatial information. As a result, when loaded into Slicer, the application will assign default spatial direction information, which may not match your data. To ensure the partial volumes retain the correct spatial information, be sure to load them via the Autoscoper Pre-Processing module with the appropriate input volume selected.
```

## Configuration File Generation

The `Generate Config` section of the Pre-Processing module an alternative way to create an [Autoscoper configuration file](../file-specifications/config.md).For more information on generating a configuration file directly within the Autoscoper application, see the section on [](./custom-data.md#creating-a-configuration-file).

The inputs you provide in this section follow the configuration file format specified in {ref}`config-file-format-version-1-1`.


### Creating a Configuration File

To begin, you will first need to specify the filename for the configuration file you wish to generate in the text field labeled `Config Trial Name`.

Next, you will select the camera calibration files and radiograph subdirectories. These paths correspond to the `mayaCam_csv` and `CameraRootDir` file format keys, respectively. Note that the order in which radiograph subdirectories are specified must match the order of the camera calibration files to ensure the correct camera is loaded.

To select the files, click the `Populate From Camera Subdirectory` button to load the contents of the `Camera Subdirectory` (as specified in the `Default Subdirectories` section) into the left-hand list. Similarly, click the `Populate From Radiographs Subdirectory` button to load the contents of the `Radiograph Subdirectory` into the left-hand list.

Once the directories are populated, you can select the files you wish to include in your configuration file. For each file, check the box to its left, and then click the green right arrow button. This will add the item to the list on the right. The order of items in the right-hand list reflects the order in which they will appear in the configuration file. It is crucial that the number of camera calibration files and radiograph subdirectories match, and that the items in both lists correspond pairwise (e.g., the first camera calibration file should correspond to the first radiograph subdirectory, the second camera calibration file to the second radiograph subdirectory, and so on).

If you wish to adjust the order of selected items, you can remove them from the right-hand list by clicking the `Delete` button next to each item. You and then re-add them in the desired order.

![Configuration File Path Selection](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/prePro_configPathSelection.gif)

For selecting partial volume files, the process is similar but without the need to specify an order. Click the `Populate from Volume Subdirectory button` to load all available partial volume files from the `Partial Volume Subdirectory`. From there, you can check the files you want to include in the configuration file.

The remaining parameters to configure are relatively straightforward: the optimization offsets, made of six decimal values; the volume flip, which can be enabled or disabled for each of the axes; the render resolution, made of two integer values (width and height); the voxel size, made of three decimal values, set automatically according to the selected input volume.

When you are done setting the necessary parameters, click the `Generate Config File` button at the bottom of this section. This will generate the configuration file and write it to the output directory specified in the `Output Directory` field at the top of the Autoscoper Pre-Processing tab.
