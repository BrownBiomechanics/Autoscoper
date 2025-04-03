# Loading Custom Data

This tutorial will show you how to prepare your own data for use with Autoscoper. This tutorial assumes that you have some familiarity with tracking data using Autoscoper. If you are new to Autoscoper, please see the [Loading and Tracking Data](./loading-and-tracking.md#tracking-a-skeletal-structure) tutorial for information on how to track data with Autoscoper.

```{note}
This tutorial is intended for users who wish to use their own data with Autoscoper. However, for learning purposes, it can still be followed using the sample data provided with Autoscoper. The sample data is downloaded into the cache directory of 3D Slicer. To locate your cache directory open 3D Slicer and go to `Edit` -> `Application Settings` -> `Cache`. You can then click on the button next to `Cache location:` to open the cache directory.
```

## Preparing the Data

Autoscoper requires the following data to track a skeletal structure:
* One or more radiograph series
* A 3D volume of each bone in the skeletal structure you wish to track
* A camera calibration file for each radiograph series

## Recommended File Structure

Below is the recommended file structure for a trial that tracks k bones over n cameras consisting of m frames each

```
trial name
├── radiographs
│   ├── camera 1
│   │   ├── 0001.tif
│   │   ├── 0002.tif
│   │   ├── ...
│   │   └── m.tif
│   └── camera 2
│   |   ├── 0001.tif
│   |   ├── 0002.tif
│   |   ├── ...
│   |   └── m.tif
|   ...
|   └── camera n
│       ├── 0001.tif
│       ├── 0002.tif
│       ├── ...
│       └── m.tif
├── volumes
│   ├── bone1.tif
│   ├── bone2.tif
│   ├── ...
│   └── bonek.tif
├── calibration
|   ├── camera 1.csv
|   ├── camera 2.csv
|   ├── ...
|   └── camera n.csv
└── trial name.cfg
```

### Automatic filter and tracking data loading

AutoscoperM can automatically load filter and tracking data from a trial. To do this, the filter and tracking data must be in the following format:

```
trial name
├── radiographs
|   └── ...
├── volumes
|   └── ...
├── calibration
|   └── ...
├── xParamters
|   └── control_settings.vie
├── Tracking
|   ├── {trial name}_bone1.tra
|   ├── {trial name}_bone2.tra
|   ├── ...
|   └── {trial name}_bonek.tra
└── trial name.cfg
```



## Camera Calibration File

For information on how to create a camera calibration file, see the [Camera Calibration File Format](../file-specifications/camera-calibration.md) page.


## Creating a Configuration File

```{note}
If you have any unsaved changes to your current trial, you will be prompted to save them before creating a new trial.
```

AutoscoperM uses a `.cfg` file to specify the location of the radiographs, volumes, and camera calibration files. You can create a `.cfg` file by going to `File` -> `New`. This will open up the configuration dialog:

![Configuration Dialog](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_ConfigDialog.png)

From the dialog you can create as many cameras as needed. Where the `Calibration File` points the the camera calibration file and the `Video Path` points to the folder containing the radiographs for each camera.

You can also create as many volumes as needed. Where the `File` points to the volume file. and the `Scale XYZ` is the scale of the volume in the x, y, and z directions. The `Scale XYZ` is used to scale the volume to match the size of the radiographs. The `   Flip XYZ` indicates whether the volume should be flipped in the x, y, and z directions. The `Flip XYZ` is used to match the orientation of the volume to the radiographs.

Once everything is set in the dialog you can press `OK` to automatically load the trial into AutoscoperM. In order to save the configuration file, you can go to `File` -> `Save Trial as`. This will save the configuration file to the specified location. You can also press `Crtl + Shift + S` to save the configuration file.
