#############
Example Usage
#############

Use PyAutoscoper to connect to Autoscoper then load and save data. 


==================
Example 1: Connect
==================

We can connect to Autoscoper using the AutoscoperConnection class. This class is a wrapper for the TCP socket connection to Autoscoper. 
The class will attempt to connect to Autoscoper upon instantiation. If the connection is successful, the is_connected method will return True.

The AutoscoperConnection class can be created with two optional arguments:
    * address: The IP address of the Autoscoper server. Default is `127.0.0.1` (localhost).
    * verbose: If True, the methods will print out information about the connection. Default is False.

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 1 - Start]
    :end-before: [Example 1 - End]
    :language: python

=============================
Example 2: Setting up a trial
=============================

We can set up a trial using the socket connection we created in Example 1. We can use the AutoscoperConnection class to load the trial configuration file.

This example will load the trial configuration file `trial_config.cfg` and the filter settings file `filter_settings.vie` for each camera.

The loadTrial method takes one argument:
    * trial_config: The path to the trial configuration file.

The loadFilters method takes two arguments:
    * camera_i: The camera index (0-indexed).
    * filter_settings: The path to the filter settings file.

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 2 - Start]
    :end-before: [Example 2 - End]
    :language: python

===========================================
Example 3: Loading and Saving Tracking Data
===========================================

We can load and save tracking data using the socket connection we created in Example 1. We can use the AutoscoperConnection class to load and save tracking data.

Note: A trial must be loaded before the loading and saving tracking data happens.

This example will load the tracking data file `tracking_data.tra` and save the tracking data to `tracking_data_out.tra`.

The loadTrackingData method takes at least two arguments:
    * volume: The volume index (0-indexed).
    * tracking_data: The path to the tracking data file.
    * is_matrix: If True, the tracking data will be loaded as a 4 by 4 matrix. If False, the tracking data will be loaded in xyz roll pitch yaw format. Defaults to True.
    * is_rows: If True, the tracking data will be loaded as rows. If False, the tracking data will be loaded as columns. Defaults to True.
    * is_with_commas: If True, the tracking data will be loaded with commas. If False, the tracking data will be loaded with spaces. Defaults to True.
    * is_cm: If True, the tracking data will be loaded in cm. If False, the tracking data will be loaded in mm. Defaults to False.
    * is_rad: If True, the tracking data will be loaded in radians. If False, the tracking data will be loaded in degrees. Defaults to False.
    * interpolate: If True, the tracking data will be interpolated with the spline method. If False, the tracking data will not be interpolated (NaN values). Defaults to False.

The saveTracking method takes at least two arguments:
    * volume: The volume index (0-indexed).
    * tracking_data: The path to the tracking data file.
    * save_as_matrix: If True, the tracking data will be saved as a 4 by 4 matrix. If False, the tracking data will be saved in xyz roll pitch yaw format. Defaults to True.
    * save_as_rows: If True, the tracking data will be saved as rows. If False, the tracking data will be saved as columns. Defaults to True.
    * save_with_commas: If True, the tracking data will be saved with commas. If False, the tracking data will be saved with spaces. Defaults to True.
    * convert_to_cm: If True, the tracking data will be saved in cm. If False, the tracking data will be saved in mm. Defaults to False.
    * convert_to_rad: If True, the tracking data will be saved in radians. If False, the tracking data will be saved in degrees. Defaults to False.
    * interpolate: If True, the tracking data will be interpolated with the spline method. If False, the tracking data will not be interpolated (NaN values). Defaults to False.

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 3 - Start]
    :end-before: [Example 3 - End]
    :language: python

===================================
Example 4: Changing Frames and Pose
===================================

We can change the frames and pose of the camera using the socket connection we created in Example 1. We can use the AutoscoperConnection class to change the frames and pose of the camera.

Note: A trial must be loaded before the frames and pose can be changed.

This example will change the pose on multiple frames of the trial.

The setPose methods takes three arguments:
    * volume: The volume index (0-indexed).
    * frame: The frame index (0-indexed).
    * pose: The pose of the camera in xyz roll pitch yaw format.

The getPose methods takes two arguments:
    * volume: The volume index (0-indexed).
    * frame: The frame index (0-indexed).

The setFrame method one argument:
    * frame: The frame index (0-indexed).

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 4 - Start]
    :end-before: [Example 4 - End]
    :language: python

========================
Example 5: Optimizations
========================

We can optimize the tracking data using the socket connection we created in Example 1. We can use the AutoscoperConnection class to optimize the tracking data.

Note: A trial must be loaded before the tracking data can be optimized.

There are two methods for optimizing the tracking data:
    * optimizeFrame: Optimizes the tracking data for a single frame.
    * trackingDialog: Automatically optimizes the tracking data for all given frames.

--------------------
optimizeFrame Method
--------------------

The optimizeFrame method takes ten arguments:
    * volume: The volume index (0-indexed).
    * frame: The frame index (0-indexed).
    * repeats: The number of times to repeat the optimization.
    * max_itr: The maximum number of iterations to run the optimization.
    * min_lim: The minimum limit for the PSO movement.
    * max_lim: The maximum limit for the PSO movement.
    * max_stall_itr: The maximum number of iterations to stall the optimization.
    * dframe: The amount of frames to skip backwards for the intial guess.
    * opt_method: The optimization method to use, 0 for Particle Swarm, 1 for Downhill Simplex.
    * cf_model: The cost function to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models)

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 5 - Start]
    :end-before: [Example 5 - End]
    :language: python

---------------------
trackingDialog Method
---------------------

The trackingDialog method takes at least three arguments:
    * volume: The volume index (0-indexed).
    * start_frame: The starting frame index (0-indexed).
    * end_frame: The ending frame index (0-indexed).
    * frame_skip: The number of frames to skip between each optimization. Defaults to 1.
    * repeats: The number of times to repeat the optimization. Defaults to 1.
    * max_itr: The maximum number of iterations to run the optimization. Defaults to 1000.
    * min_lim: The minimum limit for the PSO movement. Defaults to -3.0.
    * max_lim: The maximum limit for the PSO movement. Defaults to 3.0.
    * max_stall_itr: The maximum number of iterations to stall the optimization. Defaults to 25.
    * opt_method: The optimization method to use, 0 for Particle Swarm, 1 for Downhill Simplex. Defaults to 0.
    * cf_model: The cost function to use, 0 for NCC (Bone Models), 1 for Sum of Absolute Differences (Implant Models). Defaults to 0.

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 5.1 - Start]
    :end-before: [Example 5.1 - End]
    :language: python

=========================
Example 6: Custom Scripts
=========================

We can run custom scripts using everything we learned in the previous examples. We can use the AutoscoperConnection class to run custom scripts.

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Example 6 - Start]
    :end-before: [Example 6 - End]
    :language: python

===============================================
Bonus Example: Launching Autoscoper from Python
===============================================

We can launch Autoscoper from Python using the subprocess module. 

.. literalinclude:: ../../examples/example.py
    :caption: ./scripts/python/examples/example.py
    :start-after: [Bonus Example - Start]
    :end-before: [Bonus Example - End]
    :language: python