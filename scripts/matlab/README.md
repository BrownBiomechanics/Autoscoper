# Autoscoper MATLAB TCP Client

This document describes how to use the MATLAB TCP client to communicate with the Autoscoper server.

## History

Before these changes were made, the MATLAB TCP client was a collection of functions that were called to communicate with the server. Once the Python package, PyAutoscoper, was added to the project, it was decided that the MATLAB TCP client should be changed to a class that can mirror the functionality of the Python package. This was done to make it easier for users to switch between the two languages.

## Notes:
* The list of parameters is order dependent. Some parameters are optional, but all preceding parameters must be specified.
* The method getImageData is not implemented yet, so it will not return anything.

## Usage:

#Sample Worflow:
*launch Autoscoper
*pseudocode variables:  cfgFileName myVolumeList filterFileName trackDataInFileName  saveDataFileName sumberOfFrames
```matlab

autoscoper_socket_object = AutoscoperConnection();
aobj = autoscoper_socket_object; %short name
loadTrial(aobj, cfgFileName);
loadFilters(aobj, -1, filterFileName);

%if seeding poses available
for vNum = 0: length(myVolumeList)-1
	loadTrackingData(aobj, vNum, [trackDataInFileName,myVolumeList{vNum}]); 

	trackingDialog(vNum, tf, numberOfFrames);
	
	saveTrackingData(vNum, [saveDataFileName,myVolumeList{vNum}]);
end	
closeConnection(aobj);
```



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
OR
loadTrial(connection, path_to_cfg_file);
```

This will load a trial from the specified configuration file. The configuration file is a cfg file that contains all the information about the trial.

This function has the following parameter:

* path_to_cfg_file: The path to the configuration file.

### Load Tracking Data
  
```matlab
connection.loadTrackingData(volNum, tra_fileName, is_matrix, is_rows, is_csv, is_cm, is_rad);
OR
loadTrackingData(connection, volNum, tra_fileName, is_matrix, is_rows, is_csv, is_cm, is_rad);
```

This will load tracking data from the specified file. The tracking data file is a tra file that contains the tracking data.

This function has the following parameters:

* volNum: The volume( numeric, index 0, set by cfg order)  to load the tracking data for.
* tra_fileName: The path to the tracking data file.
* is_matrix: Optional. Whether input tracking data is matrix form. Default: true (1)  false if in xyarpy form (0)
* is_rows: Optional. Whether input tracking data is row format. Default: true (1) verified within. false(0) if in column format
* is_csv: Optional. Whether input tracking data is comma separated values. Default: true(1) false (0) if whitespace format
* is_cm: Optional. Whether input tracking data express in cm. Default: false, (interpreted as mm)
* is_rad: Optional. Whether input tracking data expressed in radians. Default: false (inter as degrees)


  ### Save Tracking Data
  
```matlab
connection.saveTrackingData(volNum, tra_fileName, save_as_matrix, save_as_rows, save_with_commas, convert_mm_to_cm, convert_deg_to_rad, interpY);
OR
saveTrackingData(connection, volNum, tra_fileName, save_as_matrix, save_as_rows, save_with_commas, convert_mm_to_cm, convert_deg_to_rad, interpY);
```

This will save the tracking data for the specified volume to the specified file. The tracking data file is a tra file that contains the tracking data.

This function has the following optional parameters:

* volNum: The volume( numeric, index 0, set by cfg order) to save the tracking data for.
* tra_fileName: The path and file name (.tra) to where tracking data will be saved
* save_as_matrix: Optional. Whether to save the tracking data as a matrix. Default: true
* save_as_rows: Optional. Whether to save the tracking data as rows. Default: true
* save_with_commas: Optional. Whether to save the tracking data with commas. Default: true
* convert_mm_to_cm: Optional. Whether to convert the tracking data to cm. Default: false
* convert_deg_to_rad: Optional. Whether to convert the tracking data to radians. Default: false
* interpY: Optional. Whether to interpolate the tracking data. Default: false



### Load Filters
  
```matlab
connection.loadFilters(camera, filter_file);
OR
loadFilters(connection, camera, filter_file);
```

This will load filters from the specified file. The filters file is a `.vie` file that contains the filter information.

This function has the following parameters:

* camera: The camera index to load the filters for. (index base 0)  -1 for all
* filter_file: The path anf fielname to the filters file. 

### Set Frame
  
```matlab
connection.setFrame(frameNum);
OR
setFrame(connection, frameNum);
```

This will set the frame to the specified frame.

This function has the following parameter:

* frameNum: The frame to set.

### Get Pose
    
```matlab
pose = connection.getPose(volNum, frameNum);
OR
pose = getPose(connection, volNum, frameNum);
```

This will get the pose for the specified volume at the specified frame.

This function has the following parameters:

* volNum: The volume to get the pose for.
* frameNum: The frame to get the pose at.

This function returns the following:

* pose: The pose for the specified volume at the specified frame.

### Set Pose
    
```matlab
connection.setPose(volume, frame, pose);
OR
setPose(connection, volume, frame, pose);
```

This will set the pose for the specified volume at the specified frame.

This function has the following parameters:

* volume: The volume to set the pose for.
* frame: The frame to set the pose at.
* pose: The pose to set.

### Get NCC
    
```matlab
ncc = connection.getNCC(voNum, pose);
OR
ncc = getNCC(connection, volNum, pose);
```

This will get the NCC for the specified volume at the specified frame.

This function has the following parameters:

* volNum: The volume to get the NCC for.
* pose: The pose to get the NCC at. (see getPose xyzrpy)

This function returns the following:
  
* ncc: The NCC for the specified volume at the specified frame.

### Set Background
    
```matlab
connection.setBackground(threshold);
OR
setBackground(connection, threshold);
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
connection.optimizeFrame(volNum, frameNum, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
OR
optimizeFrame(connection, volNum, frameNum, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
```

This will optimize the frame for the specified volume.

This function has the following parameters:

* volNum: The volume to optimize the frame for.
* frameNum: The frame to optimize.
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
connection.trackingDialog(volNum, startframe, endframe, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
OR
trackingDialog(connection, volNum, startframe, endframe, repeats, max_itr, min_lim, max_lim, max_stall_itr, dframe, opt_method, cf_model);
```

This will perform optimization for the specified volume over the specified frames. =(uses optimizeFrame)

This function has the following parameters:

* volNum: The volume to optimize the frame for.
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
ncc_sum = connection.getNCC_Sum(volNum, pose);
OR
ncc_sum = getNCC_Sum(connection, volNum, pose);
```

This will get the NCC sum for the specified volume in the specified pose (set getPose, uses getNCC)

This function has the following parameters:

* volNum: The volume to get the NCC sum for.
* pose: The pose to get the NCC sum at. 

### Get NCC This Frame

```matlab
ncc_this_frame = connection.getNCC_This_Frame(volNum, frameNum); (uses getPose, getNCC)
OR
ncc_this_frame = getNCC_This_Frame(connection, volNum, frameNum); (uses getPose, getNCC)
```

This will get the NCC this frame for the specified volume at the specified frame. (uses the current pose)

This function has the following parameters:

* volNum: The volume to get the NCC this frame for.
* frameNum: The frame to get the NCC this frame at.

*returns a three element double- the ncc values returned from getNCC  , and thier product

### Close Connection
  
```matlab
connection.closeConnection();
```

This will close the connection to the server.