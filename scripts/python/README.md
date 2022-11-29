# Autoscoper Python TCP Client

This document describes the interface allowing to interface with a running Autoscoper process from Python.

## Setup
* `cd client_connection_libs/python`
* `pip install -e .`
* Open any python file and add `from PyAutoscoper.connect import *` to the top of the file.
* Use any of the functions described below

## Available functions:

* `openConnection`
  * **Parameters:**
    * `address`
  * **Return**: autoscoper_socket
  * **MethodId**: `NA`

* `loadTrial`
  * **Parameters:**
    * `autoscoper_socket`
    * `trial_file`
  * **Return**: `None`
  * **MethodId**: `1`

* `loadTrackingData`
  * **Parameters:**
    * **Required parameters:**
        * `autoscoper_socket`
        * `volume`
        * `trackingData`
    * **Optional parameters:**
        * `save_as_matrix` - If true, the tracking data will be saved as a 4 by 4 matrix. If false, the tracking data will be saved in xyz roll pitch yaw format. Defaults to true.
        * `save_as_rows` - If true, the tracking data will be saved as rows. If false, the tracking data will be saved as columns. Defaults to true.
        * `save_with_commas` - If true, the tracking data will be saved with commas. If false, the tracking data will be saved with spaces. Defaults to true.
        * `convert_to_cm` - If true, the tracking data will be converted to cm. If false, the tracking data will be saved in mm. Defaults to false.
        * `convert_to_rad` - If true, the tracking data will be converted to radians. If false, the tracking data will be saved in degrees. Defaults to false.
        * `interpolate` - If true, the tracking data will be interpolated using the spline method. If false, the tracking data will be saved as is (with NaN values). Defaults to false.
  * **Return**: `None`
  * **MethodId**: `2`

* `saveTracking`
  * **Parameters:**
    * **Required parameters:**
        * `autoscoper_socket`
        * `volume`
        * `filename`
    * **Optional parameters:**
        * `save_as_matrix` - If true, the tracking data will be saved as a 4 by 4 matrix. If false, the tracking data will be saved in xyz roll pitch yaw format. Defaults to true.
        * `save_as_rows` - If true, the tracking data will be saved as rows. If false, the tracking data will be saved as columns. Defaults to true.
        * `save_with_commas` - If true, the tracking data will be saved with commas. If false, the tracking data will be saved with spaces. Defaults to true.
        * `convert_to_cm` - If true, the tracking data will be converted to cm. If false, the tracking data will be saved in mm. Defaults to false.
        * `convert_to_rad` - If true, the tracking data will be converted to radians. If false, the tracking data will be saved in degrees. Defaults to false.
        * `interpolate` - If true, the tracking data will be interpolated using the spline method. If false, the tracking data will be saved as is (with NaN values). Defaults to false.
  * **Return**: `None`
  * **MethodId**: `3`

* `loadFilters`
  * **Parameters:**
    * `autoscoper_socket`
    * `cameraId`
    * `filtersConfig`
  * **Return**: `None`
  * **MethodId**: `4`

* `setFrame`
  * **Parameters:**
    * `autoscoper_socket`
    * `frame`
  * **Return**: `None`
  * **MethodId**: `5`

* `getPose`
  * **Parameters:**
    * `autoscoper_socket`
    * `volume`
    * `frame`
  * **Return**: pose
  * **MethodId**: `6`

* `setPose`
  * **Parameters:**
    * `autoscoper_socket`
    * `volume`
    * `frame`
    * `pose`
  * **Return**: `None`
  * **MethodId**: `7`

* `getNCC`
  * **Parameters:**
    * `autoscoper_socket`
    * `volume`
    * `pose`
  * **Return**: `None`
  * **MethodId**: `8`

* `setBackground`
  * **Parameters:**
    * `autoscoper_socket`
    * `value`
  * **Return**: `None`
  * **MethodId**: `9`

* `optimizeFrame`
  * **Parameters:**
    * `autoscoper_socket`
    * `volumeID`
    * `frame`
    * `repeats`
    * `max_iter`
    * `min_lim`
    * `max_lim`
    * `max_stall_iter`
  * **Return**: `None`
  * **MethodId**: `11`

* `getFullDRR`
    * **Parameters:**
        * `NA`
    * **Return:** `None`
    * **MethodId:** 12
* `closeConnection`
    * **Parameters:**
        * `NA`
    * **Return:** `None`
    * **MethodId:** 13

