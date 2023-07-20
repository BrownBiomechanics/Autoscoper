# Loading and Tracking Data

This tutorial provides step-by-step instructions on how to load sample data from Slicer and track it in Autoscoper.

## Downloading Sample Data

Some short sample data is included within the SlicerAutoscoperM extension. To load this data, open Slicer and switch to the `Sample Data` module. This is located in the module drop down menu, in the top left corner of the Slicer window, under the `Informatics` section.

![Sample Data Module](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_SampleDataModule.png)

In the `Sample Data` module, scroll down the left-hand side until you see the `Tracking` section. Select your desired data by clicking on the icon for that data.

Available sample data includes:

* AutoscoperM - Wrist BVR
* AutoscoperM - Knee BVR
* AutoscoperM - Ankle BVR

![Sample Data Downloading](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_DownloadSampleData.png)

Once downloaded, switch to the `AutoscoperM` module to begin tracking. This module is located in the module drop-down menu, under the `Tracking` section.

![AutoscoperM Module](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_AutoscoperModule.png)

## Launching Autoscoper and Loading Sample Data

```{warning}
Launching Autoscoper for the first time on Windows may require you to allow the program to run.
```

Once the `AutoscoperM` module is open, click the `Launch Autoscoper` button to launch Autoscoper. This will open a new window with the `Autoscoper` interface. Once Autoscoper is open, you can load the sample data by clicking one of the buttons in the Sample Data section of the interface. The buttons are labeled `Load Wrist Data`, `Load Knee Data`, and `Load Ankle Data`.

![AutoscoperM Interface](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_LaunchAndLoad.png)

After loading the sample data, the Autoscoper window should look like this:

![Sample Data Loaded](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_SampleLoaded.png)

To zoom in on the radiographs and see the details, you can use `Control + Mouse Wheel` to zoom in and out. To adjust the position of the radiographs, you can use `Control + Left Mouse Button` to pan the radiographs.

## Tracking a Skeletal Structure

```{warning}
The filters provided with the sample data may not be the optimal filters. Since the filters play an important role in the tracking process, you may need to adjust the filters to get the best results. Please see the [](./filters.md) tutorial for more information on filters and how to adjust them.
```

### Aligning a Volume

```{warning}
The current version of AutoscoperM only supports tracking a single volume at a time. If you wish to track multiple volumes, it is recommended to align and track one volume all the way through before moving on to the next volume.
```

The first step in tracking a skeletal structure is aligning a volume to a set of bi-plane radiographs. Start by selecting the volume you wish to align from the volumes list in the lower-left corner of the screen. In this example, we will align the radius or the `rad_dcm_cropped` volume. To align the volume, move the mouse over one of the radiograph images and use the `Left Mouse Button` to move the volume around.

You can press `E` to switch to rotation mode or `W` to return to translation mode. Use `D` to move the location of the pivot point if needed. Use `S` to set a keyframe, which is used as a reference point in the tracking process. Pressing `C` will perform optimization on the current frame, which can be useful for snapping the volume to radiographs.

```{tip}
For help with aligning a volume, please see the [](./tips-and-tricks.md) tutorial.
```

![Aligned with the right radiograph](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_AlignedWithRight.png)

![Aligned with both radiographs](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_AlignedWithBoth.png)

### Tracking the volume

```{warning}
Once `OK` is pressed in the tracking dialog, the tracking process will begin. This process can take a long time, and the program will be unresponsive until the tracking is complete.

To view the output of the tracking process, you can open the Python terminal in 3D Slicer by hitting `Control + 3`. The output will be printed to the terminal.
```

Once the volume is aligned with the radiographs, you can press the `Tracking Dialog` button on the [](../user-interface.md#toolbar) to open the tracking dialog. The dialog will look like this:

![Tracking Dialog](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_TrackingDialog.png)

The dialog has several options. The first option is the `Tracking Range` which allows you to specify the range of frames you wish to track. The default is to track all the frames in order. The second option is `Initial Guess`, changing this will change how the initial position of the volume is determined. The default is to use the position of the volume in current frame. The third option is `Optimization method`, you can choose between particle swarm optimization (PSO) or downhill simplex. The default is PSO. You can also specify the number of time you want the optimization to run on each frame. The default is 1. The fourth option is `PSO Algorithm Parameters`, you can change the parameters for the optimization here. The default values are usually good enough. The last option is `Cost Function`, you can choose between the normalized cross correlation (NCC) or the sum of absolute differences (SAD). The default is NCC.

Once you have set the options, you can press the `OK` button to start tracking. The tracking will take a while to complete, and trials with lots of frames will take even longer. Once the tracking is complete, the dialog will close.

For more information see the [](../user-interface.md#tracking-dialog) section of the user interface page.

### Viewing the tracking results

Once the tracking is complete, you can view the tracking results by moving the slider at the bottom of the screen.

![Tracking Results](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_tracked.gif)

You can also view the tracking results in the 3D view by going to `View` -> `Show World View` in the main menu. You can use `Control + Middle Mouse Button` to pan the 3D view, `Control + Left Mouse Button` to rotate the 3D view, and `Control + Right Mouse Button + Drag` to zoom in and out. Move the view so you can see all of the volumes and radiographs at once. Using the slider at the bottom of the screen, you can move through the frames and see the tracking results in the 3D view.

![World View](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_WorldView.png)

### Saving the tracking results

Once you are satisfied with the tracking results, you can save the tracking results by clicking the `Save Tracking` button. This will open a dialog where you can specify the name of the file you wish to save the tracking results to. The file will be saved as a `.tra` file. This file can be loaded back into Autoscoper at a later time by clicking the `Load Tracking` button.

Once the file name is specified, you can click the `OK` button and the tracking import/export dialog will open.

```{seealso}
For more information, see [](/user-interface.md#importexport-tracking-options).
```

## Evaluating the Sample Data Tracking Results

SlicerAutoscoperM includes a module, `Tracking Evaluation`, that can be used to compare your results to some ground truth data included in the sample data.

### Exporting your Tracking Results from Autoscoper

To export your tracking results from Autoscoper, click the `Save Tracking` button on the [](../user-interface.md#toolbar). This will open a dialog where you can specify the name of the file you wish to save the tracking results to. Once you press okay, the [](../user-interface.md#importexport-tracking-options) dialog will open. Ensure that the `All` option under the `Volumes` section is selected. Then press the `OK` button to export the tracking results.

![Export Tracking Results](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_AllVolumes.png)

### Switching to the Tracking Evaluation Module

The `Tracking Evaluation` module can be found in the `tracking` category of the module drop-down menu.

![Tracking Evaluation Module](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_SwitchModule.png)

### Loading in Results

To load in your tracking results, select the file you exported from Autoscoper in the `Input Data` section of the module. Then select the sample data type you wish to compare your results to. Finally, press the `Load Data for Evaluation` button.

```{note}
You may need to adjust the camera of the 3D scene inorder to see the results clearly.
```

![Load data](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_LoadData.png)

### Visualizing Results

You can click the `Play` button in the `Visualize Results` sections to automatically scrub through the sequence. You can also use the timeline to scrub through the sequence manually. Results that are within a 1mm or 2 degree difference of the ground truth will be displayed as green. If your results exceed these thresholds, they will be displayed as red and a partially transparent version of the ground truth will be displayed.

![Results Gif](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_ShowModule.gif)
