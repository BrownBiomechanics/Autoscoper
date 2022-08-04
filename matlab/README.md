# Autoscoper Matlab TCP client

This document describes the interface allowing to interface with a running Autoscoper process from MatLab.

Notes:
* The list of **parameters** is order dependent.
* The **methodId** integer is used in [Socket::handleMessage()](https://github.com/BrownBiomechanics/Autoscoper/blob/main/autoscoper/src/net/Socket.cpp) to identifies the Autoscoper method to invoke.
* The methods `getImageData` (method id `10`) and `saveFullDRR` (method id `12`) are not available through the MatLab interface.

Available functions:

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
    * `autoscoper_socket`
    * `volume`
    * `trackingData`
  * **Return**: `None`
  * **MethodId**: `2`

* `saveTracking`
  * **Parameters:**
    * `autoscoper_socket`
    * `volume`
    * `filename`
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

* `getNCC_Sum`
  * **Parameters:**
    * `pose`
    * `autoscoper_socket`
    * `volume`
  * **Return**: `ncc_out`
  * **MethodId**: `8`

* `getNCC_This_Frame`
  * **Parameters:**
    * `autoscoper_socket`
    * `volume`
    * `frame_num`
  * **Return**: `ncc_out`
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

