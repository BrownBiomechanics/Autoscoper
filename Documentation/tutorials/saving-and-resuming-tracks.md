# Saving and Resuming a Trial

This tutorial covers the process of saving and resuming a trial, allowing users to conveniently save their progress and pick up their work later or share it with others. Before proceeding with saving and resuming a trial, make sure you have completed the [](./loading-and-tracking.md) tutorial, or have already loaded and tracked data in AutoscoperM.

## Saving a Trial

### File Structure

When saving and resuming a trial, it's essential to organize the tracks in a specific file structure to ensure automatic loading. This file structure is the same as explained in the [](./custom-data.md#automatic-filter-and-tracking-data-loading) section of the [](./custom-data.md) tutorial.


The required file structure is as follows:

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

To save a trial configuration, follow these steps:

1. Click the `File` menu and select `Save a Trial`, or press `Ctrl+Shift+S` on your keyboard.
2. A dialog box will appear, allowing you to choose a location to save the trial configuration.
3. The trial configuration will be saved as a `.cfg` file.

### Saving Tracks

To ensure tracks are automatically loaded when a trial is resumed, tracking data must be saved in the default configuration. Please see the [](../user-interface.md#importexport-tracking-options) section of the User Interface page for information on the default configuration that Autoscoper uses to save tracking data. For each bone you would like to save tracks for, follow these steps:

1. Select the bone in the `Volumes` panel (for more information, refer to the [](../user-interface.md) section).
2. Click the `File` menu and select `Save Tracking`, press `Ctrl+S` on your keyboard, or use the `Save Tracking` button on the [](../user-interface.md#toolbar).
3. A dialog box will appear, allowing you to choose a location to save the tracks.
4. The tracks will be saved as a `.tra` file in the Tracking subfolder of the trial you are working on.

### Saving Filter Settings

To automatically load filter settings when a trial is resumed, filter settings must be saved into the `xParameters` subfolder and named `control_settings.vie`. For more information on filters, refer to the [](./filters.md) tutorial.

```{note}
The current version of Autoscoper supports loading only one settings file for all cameras. If you have different filter settings for different cameras, you will need to manually load the settings for each camera after loading the trial.
```

## Resuming a Trial

To resume a trial, simply load the trial configuration file (`.cfg`)(see [](#saving-a-trial-configuration)). Follow these steps:

1. Click the `File` menu and select `Load a Trial`, press `Ctrl+O` on your keyboard, or press the `Open Trial` button on the [](../user-interface.md#toolbar).
2. A dialog box will appear, allowing you to choose the trial configuration file.
3. Select the trial configuration file and click `Open` to load the trial.

```{note}
If anything fails to load, ensure that the file structure is correct and that the files are named correctly. If the trial still fails to load, you may need to manually load the tracks and filter settings. For more information on loading tracks, refer to the [](../user-interface.md#importexport-tracking-options) section in the User Interface page. Additionally, the [](./filters.md) tutorial provides information on loading filter settings.
```
