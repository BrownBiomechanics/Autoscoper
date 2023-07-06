# Saving and Resuming a Trial

This tutorial will walk you through the process of saving and resuming a trial. This is useful if you want to save your work and resume it later, or if you want to share your work with someone else.
This tutorial assumes that you have already completed the [](./loading-and-tracking.md) tutorial or have otherwise loaded and tracked data in AutoscoperM.

## Saving a Trial

### File Structure

In order to automatically load tracks after a trial has already been loaded in AutoscoperM, the tracks must be organized using a specific file structure. This is the same file structure that is shown in the [](./custom-data.md#automatic-filter-and-tracking-data-loading) section of the [](./custom-data.md) tutorial.


The file structure is as follows:

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

### Saving a Trial Configuration

To save a trial configuration, click the `File` menu and select `Save a Trial` or press `Ctrl+Shift+S` on your keyboard. This will open a dialog box that will allow you to select a location to save the trial configuration. The trial configuration will be saved as a `.cfg` file.


### Saving Tracks

In order for tracks to be automatically loaded when a trial is loaded, tracking data must be saved in the default configuration. Please see the [](../user-interface.md#importexport-tracking-options) section of the User Interface page for information on the default configuration that Autoscoper uses to save tracking data.

For each bone you would like to save tracks for, select the bone in the `Volumes` panel (see the [](../user-interface.md) for more information). Once you have the desired bone selected, click the `File` menu and select `Save Tracking`, press `Ctrl+S` on your keyboard, or press the `Save Tracking` button on the [](../user-interface.md#toolbar). This will open a dialog box that will allow you to select a location to save the tracks. The tracks will be saved as a `.tra` file. Ensure that you are saving the tracks in the `Tracking` subfolder of the trial you are working on.

### Saving Filter Settings

In order for filter settings to be automatically loaded when a trial is loaded, filter settings must be saved into the `xParameters` subfolder and named `control_settings.vie`. For more info on filters, see the [](./filters.md) tutorial.

```{note}
The current version of Autoscoper only supports loading one settings file for all of the cameras. If you have different filter settings for different cameras, you will need to manually load the settings for each camera after loading the trial.
```

## Resuming a Trial

You can resume a trial by simply loading the trial configuration file (`.cfg`) (see [](#saving-a-trial-configuration)). To do this, click the `File` menu and select `Load a Trial`, press `Ctrl+O` on your keyboard, or press the `Open Trial` button on the [](../user-interface.md#toolbar). This will open a dialog box that will allow you to select the trial configuration file. Once you have selected the trial configuration file, click `Open` to load the trial.

```{note}
If anything fails to load, ensure that the file structure is correct and that the files are named correctly. If it still fails to load, you may need to manually load the tracks and filter settings. Please see the [](../user-interface.md#importexport-tracking-options) section of the User Interface page for more information on loading tracks and the [](./filters.md) tutorial for more information on loading filter settings.
```
