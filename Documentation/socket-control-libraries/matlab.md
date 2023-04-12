# MATLAB Socket Control

This document describes how to use the MATLAB TCP client to communicate with the Autoscoper server.

## Setup
Similar to the Python interface, the MATLAB TCP client is also a class based implementation. So the first step is to update the MATLAB path to include the directory containing the `AutoscoperConnection.m` file. This can be done by running the following command in the MATLAB command window:

```matlab
addpath /path/to/autoscoper/scripts/matlab
```

## Usage

### Establishing a Connection
The MATLAB TCP client is a class based implementation. So the first step is to create an instance of the `AutoscoperConnection` class. This can be done by running the following command in the MATLAB command window:

```matlab
conn = AutoscoperConnection();
```

This will setup a TCP connection to the default address; `127.0.0.1`. If you want to connect to a different address, you can pass the address as an argument to the constructor:

```matlab
conn = AutoscoperConnection(myAddress);
```

### Closing a Connection
To close the connection, you can call the `closeConnecation` method on the connection object:

```matlab
conn.closeConnection();
```

### Loading a Trial
To load a trial, you can call the `loadTrial` method on the connection object:

```matlab
conn.loadTrial("path/to/cfg/file.cfg");
```

### Loading Tracking Data
To load tracking data, you can call the `loadTrackingData` method on the connection object:

```matlab
conn.loadTrackingData(
    volumeNumber,
    "path/to/tracking/file.tra"
)
```

There are also several optional arguments that can be passed to the `loadTrackingData` method:

```matlab
conn.loadTrackingData(
    volumeNumber,
    "path/to/tracking/file.tra",
    is_matrix,
    is_rows,
    is_csv,
    is_cm,
    is_rad,
    interpY
)
```

All of these arguments are optional and have default values. The default values are as follows:

| Argument | Default Value | Description |
| -------- | ------------- | ----------- |
| is_matrix | true | If true, the tracking data will be loaded as a 4x4 matrix. If false, the tracking data will be loaded as a 6x1 vector. |
| is_rows | true | If true, the tracking data will be loaded as a row vector. If false, the tracking data will be loaded as a column vector. |
| is_csv | true | If true, the tracking data will be loaded as comma separated values. If false, the tracking data will be loaded as space separated values. |
| is_cm | false | If true, the tracking data will be converted from millimeters to centimeters. |
| is_rad | false | If true, the tracking data will be converted from degrees to radians. |
| interpY | false | If true, the tracking data will be interpolated in the Y direction using spline interpolation. |

### Saving Tracking Data
To save tracking data, you can call the `saveTrackingData` method on the connection object:

```matlab
conn.saveTrackingData(
    volumeNumber,
    "path/to/tracking/file.tra"
)
```

There are also several optional arguments that can be passed to the `saveTrackingData` method:

```matlab
conn.saveTrackingData(
    volumeNumber,
    "path/to/tracking/file.tra",
    save_as_matrix,
    save_as_rows,
    save_as_csv,
    convert_mm_to_cm,
    convert_deg_to_rad,
    interpY
)
```

All of these arguments are optional and have default values. The default values are as follows:

| Argument | Default Value | Description |
| -------- | ------------- | ----------- |
| save_as_matrix | true | If true, the tracking data will be saved as a 4x4 matrix. If false, the tracking data will be saved as a 6x1 vector. |
| save_as_rows | true | If true, the tracking data will be saved as a row vector. If false, the tracking data will be saved as a column vector. |
| save_as_csv | true | If true, the tracking data will be saved as comma separated values. If false, the tracking data will be saved as space separated values. |
| convert_mm_to_cm | false | If true, the tracking data will be converted from millimeters to centimeters. |
| convert_deg_to_rad | false | If true, the tracking data will be converted from degrees to radians. |
| interpY | false | If true, the tracking data will be interpolated in the Y direction using spline interpolation. |

### Loading Filters
To load filters, you can call the `loadFilters` method on the connection object:

```matlab
conn.loadFilters(cameraNum,"path/to/filters/file.vie");
```

### Setting Current Frame
To set the current frame, you can call the `setFrame` method on the connection object:

```matlab
conn.setFrame(frameNumber);
```

### Getting the Pose 
To get the pose of a volume on a frame, you can call the `getPose` method on the connection object:

```matlab
pose = conn.getPose(volumeNumber,frameNumber);
```

This will return an array of length 6 containing the x, y, z, roll, pitch, and yaw of the volume on the specified frame.

### Setting the Pose
To set the pose of a volume on a frame, you can call the `setPose` method on the connection object:

```matlab
conn.setPose(volumeNumber,frameNumber,pose);
```

### Getting the NCC Value
To get the NCC value of a volume's pose, you can call the `getNCC` method on the connection object:

```matlab
ncc = conn.getNCC(volumeNumber,pose);
```

To get the sum of the NCC values of a volume's pose, you can call the `getNCC_Sum` method on the connection object:

```matlab
ncc_sum = conn.getNCC_Sum(volumeNumber,pose);
```

To get the NCC value of a volume from a frame, you can call the `getNCC_This_Frame` method on the connection object:

```matlab
ncc_this_frame = conn.getNCC_This_Frame(volumeNumber,frameNumber);
```

### Updating the Background Threshold
To update the background threshold, you can call the `setBackground` method on the connection object:

```matlab
conn.setBackground(threshold);
```

### Getting the Cropped Image
**WARNING: This method is not fully implemented yet.**

To get the cropped image of a volume on a frame, you can call the `getCroppedImage` method on the connection object:

```matlab
croppedImage = conn.getCroppedImage(volumeNumber,cameraNumber,frameNumber);
```

### Optimize Frame
To optimize the position of a volume on a frame, you can call the `optimizeFrame` method on the connection object:

```matlab
conn.optimizeFrame(volumeNumber,frameNumber);
```

There are also many optional arguments that can be passed to the `optimizeFrame` method:

```matlab
conn.optimizeFrame(
    volumeNumber,
    frameNumber,
    repeats,
    max_itr,
    min_lim,
    max_lim,
    max_stall_itr,
    dframe,
    opt_method,
    cf_model
)
```

All of these arguments are optional and have default values. The default values are as follows:

| Argument | Default Value | Description |
| -------- | ------------- | ----------- |
| repeats | 1 | Number of times to repeat the optimization. |
| max_itr | 1000 | Maximum number of iterations to run the optimization. |
| min_lim | -3.0 | Minimum limit for the Partial Swarm Optimization to move the volume |
| max_lim | 3.0 | Maximum limit for the Partial Swarm Optimization to move the volume |
| max_stall_itr | 25 | Maximum number of iterations to run the optimization without improvement. |
| dframe | 1 | The amount of frames to skip over. | 
| opt_method | 0 | The optimization method to use. 0 is Partial Swarm Optimization and 1 is Downhill Simplex. |
| cf_model | 0 | The cost function model to use. 0 is NCC and 1 is Sum of Absolute Differences. |

### Tracking Dialog
To emulate the tracking dialog from the GUI, you can call the `trackingDialog` method on the connection object:

```matlab
conn.trackingDialog(
    volumeNumber,
    frameNumber,
    startFrame,
    endFrame
);
```

The tracking dialog method has the same optional arguments as the `optimizeFrame` method.

### Save Full DRR Image
To save the full DRR image of the scene, you can call the `saveFullDRRImage` method on the connection object:

```matlab
conn.saveFullDRR()
```