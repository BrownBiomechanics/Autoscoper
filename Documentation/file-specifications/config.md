# Configuration File Format

## Version 1.1

### Overview

The configuration file is a CFG file that contains all of the information to load a trial into Autoscoper. The configuration file is used to load camera calibration files, videos, and volumes. The file is organized with key value pairs and supports relative paths. The configuration file is loaded by Autoscoper when the user selects a trial to load.

### Key value pairs

| Key | Value |
| --- | --- |
| `Version` | The version of the configuration file. |
| `mayaCam_csv` | The path to the camera calibration file. See more information about the camera calibration file format [here](camera-calibration.md). |
| `CameraRootDir` | The root directory for the radiographs. The order must be the same as the order of the `mayaCam_csv` keys to ensure the correct camera is loaded. |
| `VolumeFile` | The path to the volume file. |
| `VolumeFlip` | The flip of the volume. |
| `VoxelSize` | The voxel size of the volume. |
| `RenderingResolution` | The resolution of the 2D rendering. |
| `OptimizationOffsets` | The offsets for the optimization. |

### Example

Given a directory structure like this:

```
Main Directory
├───Calibration
│   ├───cam01.csv
│   └───cam02.csv
├───RadiographImages
│   ├───cam01
│   │   ├───0001.tif
│   │   ├───0002.tif
│   │   ├───...
│   │   └───n.tif
│   └───cam02
│       ├───0001.tif
│       ├───0002.tif
│       ├───...
│       └───n.tif
├───VolumeImages
│   └───radius.tif
└───config.cfg
```

The config file would be placed in the `Main Directory` and would look like this:


```{note}
All relative paths must be in the format `path/to/something/` or `./path/to/something/`.

Any relative paths that are in the format `/path/to/something/` will fail to load properly.
```

```
# This is a comment
Version 1.1

mayaCam_csv Calibration/cam01.csv
mayaCam_csv Calibration/cam02.csv

CameraRootDir RadiographImages/cam01/
CameraRootDir RadiographImages/cam02/

VolumeFile VolumeImages/radius.tif
VolumeFlip 0 0 0
VoxelSize 0.1 0.1 0.1

RenderingResolution 512 512

OptimizationOffsets 0.1 0.1 0.1 0.1 0.1 0.1
```

## Version 1.0

### Overview

```{warning}
This version of the configuration file does not support relative paths and is not recommended for use.
```

The configuration file is a CFG file that contains all of the information to load a trial into Autoscoper. The configuration file is used to load camera calibration files, videos, and volumes. The file is organized with key value pairs and does not support relative paths. The configuration file is loaded by Autoscoper when the user selects a trial to load.

### Key value pairs

| Key | Value |
| --- | --- |
| `mayaCam_csv` | The path to the camera calibration file. See more information about the camera calibration file format [here](camera-calibration.md). |
| `CameraRootDir` | The root directory for the radiographs. The order must be the same as the order of the `mayaCam_csv` keys to ensure the correct camera is loaded. |
| `VolumeFile` | The path to the volume file. |
| `VolumeFlip` | The flip of the volume. |
| `VoxelSize` | The voxel size of the volume. |
| `RenderingResolution` | The resolution of the 2D rendering. |
| `OptimizationOffsets` | The offsets for the optimization. |



### Example

```
# This is a comment
mayaCam_csv C:/Users/username/Documents/Autoscoper/Calibration/cam01.csv
mayaCam_csv C:/Users/username/Documents/Autoscoper/Calibration/cam02.csv

CameraRootDir C:/Users/username/Documents/Autoscoper/RadiographImages/cam01/
CameraRootDir C:/Users/username/Documents/Autoscoper/RadiographImages/cam02/

VolumeFile C:/Users/username/Documents/Autoscoper/VolumeImages/radius.tif
VolumeFlip 0 0 0
VoxelSize 0.1 0.1 0.1

RenderingResolution 512 512

OptimizationOffsets 0.1 0.1 0.1 0.1 0.1 0.1
```