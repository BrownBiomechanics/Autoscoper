# User Interface

## Overview

The main window is broken up into 6 main sections:
* Menu Bar
* Toolbar
* 2D Viewer
* Rendering Options
* Volume Selection
* Timeline

![Main Window](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_Overview.png)

## Menu Bar

The menu bar contains the following menus:
* File
    * Contains options for:
        * Trial I/O: Opening, saving, and creating new trials
        * Tracking data I/O: Importing and exporting tracking data
* Edit
    * Contains options for:
        * Setting a new threshold for the background
        * Basic utiliy functions such as undo, redo, and copy/paste
* Tracking
    * Contains options for:
        * Tracking data I/O: Importing and exporting tracking data
        * Inserting Keyframes
        * Smoothing the timeline
* Export
    * Contains options for exporting NCC values.
* Help
    * Contains the about and sample data.
* View
    * Contains options for:
        * Showing the 3D world view
        * Editing the layout of the 2D viewer

### New Trial Dialog

![New Trial Dialog](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_ConfigDialog.png)

The New Trial Dialog has 2 sections that allow the user to add as many cameras and volumes as they want.

#### Cameras

The Cameras section contains the following options:
* Calibration File
    * Opens a file dialog to select a `.txt` calibration file.
    * For more information on the calibration file format, see the [Calibration File Format Specification](file-specifications\camera-calibration.md).
* Video Path
    * Opens a file dialog to select a directory containing the corresponding tiff sequences for the calibration file.

#### Volumes

The Volumes section contains the following options:
* Volume File
    * Opens a file dialog to select a `.tif` volume file.
* Scale XYZ
    * Allows the user to set the scale of the volume in the x, y, and z directions.
* Flip XYZ
    * Allows the user to flip the volume in the x, y, and z directions.

### 3D World View

![3D World View](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_WorldView.png)

The 3D world view is a 3D representation of the volumes and cameras in the trial. The user can use `Control + Middle Mouse Button` to pan the 3D view, `Control + Left Mouse Button to rotate the 3D view`, and `Control + Right Mouse Button + Drag` to zoom in and out.


## Toolbar

![Toolbar](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_Toolbar.png)

The tool bar is broken up into 3 sections:
* File I/O
* Volume Manipulation
* Tracking

### File I/O

The file I/O section contains the following buttons:
* Open Trial
    * Opens a file dialog to select a `.cfg` configuration file.
* Save Tracking
    * Opens a file dialog to select a `.tra` tracking file, then opens an Import/Export Tracking Options dialog.
* Load Tracking
    * Opens a file dialog to select a `.tra` tracking file, then opens an Import/Export Tracking Options dialog.

#### Import/Export Tracking Options

![Import/Export Tracking Options](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_TrackingIODialog.png)

The Import/Export Tracking Options dialog has the following options:
* Volumes: Current or All
    * Selects whether to import/export the current volume or all volumes.
* Type: Matrix or xyzypr
    * Selects whether to import/export the tracking data as a 4x4 matrix or as a list of x, y, z, yaw, pitch, and roll values.
* Orientation: Row or Column
    * Selects whether to import/export the tracking data as a row or column vector.
* Seperator: Comma or Space
    * Selects whether to import/export the tracking data as a comma or space seperated list.
* Interpolation: None or Spline
    * Selects whether to import/export the tracking data as is or to interpolate the data using spline interpolation.
* Translation Units: mm or cm
    * Selects whether to import/export the translation data in millimeters or centimeters.
* Rotation Units: Degrees or Radians
    * Selects whether to import/export the rotation data in degrees or radians.

### Volume Manipulation

The volume manipulation section contains the following buttons:
* Translate
    * When pressed, the user can click and drag the mouse to translate the volume. The user can also use the shortcut `W` to toggle this mode.
* Rotate
    * When pressed, the user can click and drag the mouse to rotate the volume. The user can also use the shortcut `E` to toggle this mode.
* Move Pivot
    * When pressed, the user can click and drag the mouse to move the pivot point of the volume. The user can also use the shortcut `D` to toggle this mode.

### Tracking

The tracking section contains the following buttons:
* Tracking Dialog
    * Opens the Tracking Dialog.
* Track Current
    * Tracks the current frame. The user can also use the shortcut `C` to track the current frame.

#### Tracking Dialog

![Tracking Dialog](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_TrackingDialog.png)

The Tracking Dialog has the following options:
* Tracking Range
    * The range of frames to track and the number of frames to skip between tracking.
    * Reverse Tracking
        * If checked, the tracking will be done in reverse.
* Inital Guess:
    * Current frame: It uses the position of the “bone” in the current frame as the initial position
    * Previous frame: It uses the position of the “bone” in the previous frame as the initial position
    * Linear extrapolation: It estimates the initial position using a linear extrapolation of the previous two frames.
    * Spline interpolation: It estimates the initial position using a spline interpolation of all frames. [This is the curve that you see in the Timeline Window (bottom)]
* Optimization Method:
    * Partical Swarm Optimization (PSO): Global Minimization Algorithm. This method takes longer time, but the initialization does not matter as much as it matters for other methods
    * Downhill Simplex (DS): Fast Local Minimization Algorithm. Initialization is really important in this method.
    * Number of refinements: Number of times the optimization algorithm looks for the best match. This does not matter for PSO; however, it improves the Downhill Simplex (rule of thumb is 10 for DS).
* PSO Options
    * Min limit: This assigns a minimum neighborhood that PSO looks for the best match. Default is -3, which means PSO looks for the best match in the neighborhood of 3 mm and 3 degree of the initial position.
    * Max limit: This assigns a maximum neighborhood that PSO looks for the best match. Default is +3, which means PSO looks for the best match in the neighborhood of 3 mm and 3 degree of the initial position.
    * Max Epochs: How many epochs you want the optimization to run before it stops. Default is 1000, however, it is unlikely that it will reach this number.
    * Max Stall: Stopping criteria for PSO. If the best match does not change for this number of epochs, the optimization stops. Default is 25.
* Cost Function:
    * Normalized Cross Correlation (NCC): A normalzied cost function to detect the best match. The closer to 0 the better the match. However, this is dependent on the image filters and qualities.
    * Sum of Absolute Differences (SAD): A cost function to detect the best match. The closer to 0 the better the match. However, this is dependent on the image filters and qualities.

## 2D Viewer

![2D Viewer](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_2Dviewer.png)

The 2D viewer is where the user can view and interact with the volumes and cameras in the trial. Clicking on the scene will allow the user to interact with the volumes. To adjust the view, the user can use the following shortcuts:
* `Control + Left Mouse Button + Drag` to pan the view.
* `Control + Scroll Wheel` to zoom in and out.

### Rendering Option

![Rendering Options](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_RenderingOptions.png)

This section allows the user to adjust the rendering options for the 2D viewer.

For each camera in the trial the user can adjust the following options:
* Rad Renderer: Toggles the radiograph rendering for that camera.
* DRR Renderer: Toggles the rendering of the volume for that camera.

The user can also add filters to either the radiograph or DRR renderers (Right click on the desired renderer). The following filters are available:
* Sobel
* Contrast
* Gaussian
* Sharpen

The user can toggle the filters on and off by clicking the checkbox or adjust the parameters of the filter by clicking the wrench button.
Right clicking on a camera allows the user to save or load the camera's filter settings.

### Volume Selection

![Volume Selection](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_VolumeSelector.png)

This section allows the user to select which volume is currently active for manipulation and tracking.

### Timeline

![Timeline](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/UI_Timeline.png)

This section allows the user to scrub through the frames of the trial. And view the X, Y, Z, Yaw, Pitch, and Roll values for the selected volume.
