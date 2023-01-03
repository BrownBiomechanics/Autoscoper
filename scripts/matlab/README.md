# Autoscoper MATLAB TCP Client

This document describes how to use the MATLAB TCP client to communicate with the Autoscoper server.

## History

Before these changes were made, the MATLAB TCP client was a collection of functions that were called to communicate with the server. Once the Python package, PyAutoscoper, was added to the project, it was decided that the MATLAB TCP client should be changed to a class that can mirror the functionality of the Python package. This was done to make it easier for users to switch between the two languages.

## Notes:
* The list of parameters is order dependent. Some parameters are optional, but all preceding parameters must be specified.
* The method getImageData is not implemented yet, so it will not return anything.

## Usage:

### Connect to the server & Object creation

```matlab
connection = AutoscoperConnection();
```

This will create a connection object that can be used to communicate with the server. The connection object will be used to communicate with the server.

This function has the following parameter:

* address: Optional. The address of the server. Default: 127.0.0.1

### Load a Trial

```matlab
connection.loadTrial(path_to_cfg_file);
```

This will load a trial from the specified configuration file. The configuration file is a cfg file that contains all the information about the trial.

This function has the following parameter:

* trial: The path to the configuration file.

### Load Tracking Data
  
```matlab
connection.loadTrackingData(volume, tracking_data, is_matrix, is_rows, is_with_commas, is_cm, is_rad, interpolate);
```

This will load tracking data from the specified file. The tracking data file is a tra file that contains the tracking data.

This function has the following parameters:

* volume: The volume to load the tracking data for.
* tracking_data: The path to the tracking data file.
* is_matrix: Optional. Whether to save the tracking data as a matrix. Default: true
* is_rows: Optional. Whether to save the tracking data as rows. Default: true
* is_with_commas: Optional. Whether to save the tracking data with commas. Default: true
* is_cm: Optional. Whether to convert the tracking data to cm. Default: false
* is_rad: Optional. Whether to convert the tracking data to radians. Default: false
* interpolate: Optional. Whether to interpolate the tracking data. Default: false

  ### Save Tracking Data
  
```matlab
connection.saveTrackingData(volume, tracking_data, save_as_matrix, save_as_rows, save_with_commas, convert_to_cm, convert_to_rad, interpolate);
```

This will save the tracking data for the specified volume to the specified file. The tracking data file is a tra file that contains the tracking data.

This function has the following parameters:

* volume: The volume to save the tracking data for.
* tracking_data: The path to the tracking data file.
* save_as_matrix: Optional. Whether to save the tracking data as a matrix. Default: true
* save_as_rows: Optional. Whether to save the tracking data as rows. Default: true
* save_with_commas: Optional. Whether to save the tracking data with commas. Default: true
* convert_to_cm: Optional. Whether to convert the tracking data to cm. Default: false
* convert_to_rad: Optional. Whether to convert the tracking data to radians. Default: false
* interpolate: Optional. Whether to interpolate the tracking data. Default: false

### Load Filters
  
```matlab
connection.loadFilters(camera, filter_file);
```

This will load filters from the specified file. The filters file is a `.vie` file that contains the filter information.

This function has the following parameters:

* camera: The camera to load the filters for.
* filter_file: The path to the filters file.

### Set Frame
  
```matlab
connection.setFrame(frame);
```

This will set the frame to the specified frame.

This function has the following parameter:

* frame: The frame to set.

### Get Pose
    
```matlab
pose = connection.getPose(volume, frame);
```

This will get the pose for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to get the pose for.
* frame: The frame to get the pose at.

This function returns the following:

* pose: The pose for the specified volume at the specified frame.

### Set Pose
    
```matlab
connection.setPose(volume, frame, pose);
```

This will set the pose for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to set the pose for.
* frame: The frame to set the pose at.
* pose: The pose to set.

### Get NCC
    
```matlab
ncc = connection.getNCC(volume, frame);
```

This will get the NCC for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to get the NCC for.
* frame: The frame to get the NCC at.

This function returns the following:
  
* ncc: The NCC for the specified volume at the specified frame.

### Set Background
    
```matlab
connection.setBackground(threshold);
```

This will set the background threshold.

This function has the following parameter:

* threshold: The threshold to set.

### Get Image Cropped
    
```matlab
connection.getImageCropped(volume, camera, pose);
```

**THIS FUNCTION IS NOT IMPLEMENTED YET.**

This will get the image cropped for the specified volume, camera and pose.

This function has the following parameters:

* volume: The volume to get the image cropped for.
* camera: The camera to get the image cropped for.
* pose: The pose to get the image cropped at.

### Optimize Frame

```matlab
connection.optimizeFrame(volume, frame, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
```

This will optimize the frame for the specified volume.

This function has the following parameters:

* volume: The volume to optimize the frame for.
* frame: The frame to optimize.
* repeats: Optional. The number of times to repeat the optimization. Default: 1
* max_itr: Optional. The maximum number of iterations. Default: 1000
* min_lim: Optional. The minimum limit. Default: -3.0
* max_lim: Optional. The maximum limit. Default: 3.0
* max_stall_itr: Optional. The maximum number of iterations to stall. Default: 25
* dframe: Optional. The number of frames to skip. Default: 1
* opt_method: Optional. The optimization method,0 for Particle Swarm, 1 for Downhill Simplex. Default: 0
* cf_model: Optional. The cost function model, 0 for NCC, 1 for Sum of Absolute Differences. Default: 0

### Tracking Dialog
  
```matlab
connection.trackingDialog(volume, startframe, endframe, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
```

This will perform optimization for the specified volume over the specified frames.

This function has the following parameters:

* volume: The volume to optimize the frame for.
* startframe: The start frame to optimize.
* endframe: The end frame to optimize.
* repeats: Optional. The number of times to repeat the optimization. Default: 1
* max_itr: Optional. The maximum number of iterations. Default: 1000
* min_lim: Optional. The minimum limit. Default: -3.0
* max_lim: Optional. The maximum limit. Default: 3.0
* max_stall_itr: Optional. The maximum number of iterations to stall. Default: 25
* dframe: Optional. The number of frames to skip. Default: 1
* opt_method: Optional. The optimization method,0 for Particle Swarm, 1 for Downhill Simplex. Default: 0
* cf_model: Optional. The cost function model, 0 for NCC, 1 for Sum of Absolute Differences. Default: 0

### Get NCC Sum
    
```matlab
ncc_sum = connection.getNCCSum(volume, frame);
```

This will get the NCC sum for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to get the NCC sum for.
* frame: The frame to get the NCC sum at.

### Get NCC This Frame

```matlab
ncc_this_frame = connection.getNCCThisFrame(volume, frame);
```

This will get the NCC this frame for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to get the NCC this frame for.
* frame: The frame to get the NCC this frame at.

### Close Connection
  
```matlab
connection.closeConnection();
```

This will close the connection to the server.