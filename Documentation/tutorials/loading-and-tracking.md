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

### Aligning a Volume

The first step in tracking a skeletal structure is aligning a volume to a set of bi-plane radiographs. Start by selecting the volume you wish to align from the volumes list in the lower-left corner of the screen. In this example, we will align the radius or the `rad_dcm_cropped` volume. To align the volume, move the mouse over one of the radiograph images and use the `Left Mouse Button` to move the volume around.

You can press `E` to switch to rotation mode or `W` to return to translation mode. Use `D` to move the location of the pivot point if needed. Use `S` to set a keyframe, which is used as a reference point in the tracking process. Pressing `C` will perform optimization on the current frame, which can be useful for snapping the volume to radiographs.

```{tip}
It may help if you align the volume with one of the radiographs first, then ensure that the volume aligns with all the radiographs.
```

![Aligned with the right radiograph](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_AlignedWithRight.png)

![Aligned with both radiographs](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_AlignedWithBoth.png)

### Tracking the volume

```{warning}
Once `OK` is pressed in the tracking dialog, the tracking process will begin. This process can take a long time, and the program will be unresponsive until the tracking is complete.

To view the output of the tracking process, you can open the Python terminal in 3D Slicer by hitting `Control + 3`. The output will be printed to the terminal.
```

Once the volume is aligned with the radiographs, you can press the `Tracking Dialog` button to open the tracking dialog. The dialog will look like this:

![Tracking Dialog](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_TrackingDialog.png)

The dialog has several options. The first option is the `Tracking Range` which allows you to specify the range of frames you wish to track. The default is to track all the frames in order. The second option is `Initial Guess`, changing this will change how the initial position of the volume is determined. The default is to use the position of the volume in current frame. The third option is `Optimization method`, you can choose between particle swarm optimization (PSO) or downhill simplex. The default is PSO. You can also specify the number of time you want the optimization to run on each frame. The default is 1. The fourth option is `PSO Algorithm Parameters`, you can change the parameters for the optimization here. The default values are usually good enough. The last option is `Cost Function`, you can choose between the normalized cross correlation (NCC) or the sum of absolute differences (SAD). The default is NCC.

Once you have set the options, you can press the `OK` button to start tracking. The tracking will take a while to complete, and trials with lots of frames will take even longer. Once the tracking is complete, the dialog will close.

### Viewing the tracking results

Once the tracking is complete, you can view the tracking results by moving the slider at the bottom of the screen.

![Tracking Results](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_tracked.gif)

You can also view the tracking results in the 3D view by going to `View` -> `Show World View` in the main menu. You can use `Control + Middle Mouse Button` to pan the 3D view, `Control + Left Mouse Button` to rotate the 3D view, and `Control + Right Mouse Button + Drag` to zoom in and out.

![World View](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_WorldView.png)

### Saving the tracking results

Once you are satisfied with the tracking results, you can save the tracking results by clicking the `Save Tracking` button. This will open a dialog where you can specify the name of the file you wish to save the tracking results to. The file will be saved as a `.tra` file. This file can be loaded back into Autoscoper at a later time by clicking the `Load Tracking` button.

Once the file name is specified, you can click the `OK` button and the tracking import/export dialog will open.

```{seealso}
For more information, see [](/user-interface.md#importexport-tracking-options).
```
