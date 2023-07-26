# PyAutoscoper

## Overview

PyAutoscoper is a Python library for controlling Autoscoper via a TCP socket connection. It provides a simple interface for sending commands and receiving responses from Autoscoper.

## Installation

Install the latest version of PyAutoscoper from PyPI using pip:

```bash
$ pip install pyautoscoper
```

## Usage

### Connecting to Autoscoper

The AutoscoperConnection class can be created with two optional arguments:
* `address`: The IP address of the Autoscoper server. Default is `127.0.0.1` (localhost).
* `verbose`: If True, the methods will print out information about the connection. Default is False.

Ensure that Autoscoper is running, the class will attempt to connect to Autoscoper upon instantiation. If the connection is successful, the `is_connected` method will return True.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Connecting to Autoscoper
:start-after: "[Example 1 - Start]"
:end-before: "[Example 1 - End]"
```

### Loading a trial

This example will load the trial configuration file `trial_config.cfg` and the filter settings file `filter_settings.vie` for both cameras.

The `loadTrial` method takes one argument:
* `trial_config`: The path to the trial configuration file.

The `loadFilter` method takes two arguments:
* `camera`: The camera number (0-indexed).
* `filter_file`: The path to the filter settings file.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Loading a trial
:start-after: "[Example 2 - Start]"
:end-before: "[Example 2 - End]"
```

### Loading and Saving Tracking Data

This example will load the tracking data file `tracking_data.tra` and save it as `tracking_data_out.tra`.

```{note}
A trial must be loaded before loading tracking data.
```

The `loadTrackingData` method takes at least two arguments:
* `volume`: The volume index (0-indexed).
* `tracking_data`: The path to the tracking data file.
* `is_matrix`: If True, the tracking data will be loaded as a 4 by 4 matrix. If False, the tracking data will be loaded in xyz roll pitch yaw format. Defaults to True.
* `is_rows`: If True, the tracking data will be loaded as rows. If False, the tracking data will be loaded as columns. Defaults to True.
* `is_with_commas`: If True, the tracking data will be loaded with commas. If False, the tracking data will be loaded with spaces. Defaults to True.
* `is_cm`: If True, the tracking data will be loaded in cm. If False, the tracking data will be loaded in mm. Defaults to False.
* `is_rad`: If True, the tracking data will be loaded in radians. If False, the tracking data will be loaded in degrees. Defaults to False.
* `interpolate`: If True, the tracking data will be interpolated with the spline method. If False, the tracking data will not be interpolated (NaN values). Defaults to False.

The `saveTrackingData` method takes at least two arguments:
* `volume`: The volume index (0-indexed).
* `tracking_data`: The path to the tracking data file.
* `save_as_matrix`: If True, the tracking data will be saved as a 4 by 4 matrix. If False, the tracking data will be saved in xyz roll pitch yaw format. Defaults to True.
* `save_as_rows`: If True, the tracking data will be saved as rows. If False, the tracking data will be saved as columns. Defaults to True.
* `save_with_commas`: If True, the tracking data will be saved with commas. If False, the tracking data will be saved with spaces. Defaults to True.
* `convert_to_cm`: If True, the tracking data will be saved in cm. If False, the tracking data will be saved in mm. Defaults to False.
* `convert_to_rad`: If True, the tracking data will be saved in radians. If False, the tracking data will be saved in degrees. Defaults to False.
* `interpolate`: If True, the tracking data will be interpolated with the spline method. If False, the tracking data will not be interpolated (NaN values). Defaults to False.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Loading and Saving Tracking Data
:start-after: "[Example 3 - Start]"
:end-before: "[Example 3 - End]"
```

### Changing the Current Frame and Pose

This example will change the pose on multiple frames.

```{note}
 A trial must be loaded before changing the current frame and pose.
```

The `setPose` method takes three arguments:
* `volume`: The volume index (0-indexed).
* `frame`: The frame index (0-indexed).
* `pose`: The pose of the volume in the form of an array of 6 floats. Array order is [x, y, z, roll, pitch, yaw].

The `getPose` method takes two arguments:
* `volume`: The volume index (0-indexed).
* `frame`: The frame index (0-indexed).

The `setFrame` method takes one argument:
* `frame`: The frame index (0-indexed).

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Changing the Current Frame and Pose
:start-after: "[Example 4 - Start]"
:end-before: "[Example 4 - End]"
```

### Optimizations

There are two methods for optimizing the tracking data:
* `optimizeFrame`: Optimizes the tracking data for a single frame.
* `trackingDialog`: Automatically optimizes the tracking data for all given frames.

The `optimizeFrame` method takes ten arguments:
* `volume`: The volume index (0-indexed).
* `frame`: The frame index (0-indexed).
* `repeats`: The number of times to repeat the optimization.
* `max_itr`: The maximum number of iterations to run the optimization.
* `min_lim`: The minimum limit for the PSO movement.
* `max_lim`: The maximum limit for the PSO movement.
* `max_stall_itr`: The maximum number of iterations to stall the optimization.
* `dframe`: The amount of frames to skip backwards for the initial guess.
* `opt_method`: The {const}`~PyAutoscoper.connect.OptimizationMethod` to use.
* `cf_model` : The {const}`~PyAutoscoper.connect.CostFunction` to use for evaluating the optimization.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Optimizing a single frame
:start-after: "[Example 5 - Start]"
:end-before: "[Example 5 - End]"
```

The `trackingDialog` method takes at least three arguments:
* `volume`: The volume index (0-indexed).
* `start_frame`: The starting frame index (0-indexed).
* `end_frame`: The ending frame index (0-indexed).
* `frame_skip`: The number of frames to skip between each optimization. Defaults to 1.
* `repeats`: The number of times to repeat the optimization. Defaults to 1.
* `max_itr`: The maximum number of iterations to run the optimization. Defaults to 1000.
* `min_lim`: The minimum limit for the PSO movement. Defaults to -3.0.
* `max_lim`: The maximum limit for the PSO movement. Defaults to 3.0.
* `max_stall_itr`: The maximum number of iterations to stall the optimization. Defaults to 25.
* `opt_method`: The {const}`~PyAutoscoper.connect.OptimizationMethod` to use. Defaults to {const}`~PyAutoscoper.connect.OptimizationMethod.PARTICAL_SWARM_OPTIMIZATION`.
* `cf_model` : The {const}`~PyAutoscoper.connect.CostFunction` to use for evaluating the optimization. Defaults to {const}`~PyAutoscoper.connect.CostFunction.NORMALIZED_CROSS_CORRELATION`.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Optimizing multiple frames
:start-after: "[Example 5.1 - Start]"
:end-before: "[Example 5.1 - End]"
```

### Putting it all together

We can put all of the above examples together to create a script that will load a trial, load tracking data, optimize the tracking data, and save the tracking data.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Putting it all together
:start-after: "[Example 6 - Start]"
:end-before: "[Example 6 - End]"
```

### Launching Autoscoper from Python

It may be useful to launch Autoscoper from Python. This can be done by using the subprocess module.

```{literalinclude} ../../scripts/python/examples/pyautoscoper-examples.py
:language: python
:caption: Launching Autoscoper from Python
:start-after: "[Example 7 - Start]"
:end-before: "[Example 7 - End]"
```

## Class Reference

```{eval-rst}
.. automodule:: PyAutoscoper.connect
   :members:
   :undoc-members:
   :show-inheritance:
```

