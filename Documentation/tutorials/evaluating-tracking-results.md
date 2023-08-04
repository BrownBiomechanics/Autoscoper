# Evaluating Tracking Results

After tracking your data in Autoscoper, you can use the `Tracking Evaluation` module in SlicerAutoscoperM to compare your results with the ground truth data included in the sample data.

## Exporting your Tracking Results from Autoscoper

To export your tracking results from Autoscoper, follow these steps:

1. Click the `Save Tracking` button on the [](../user-interface.md#toolbar).
2. In the dialog that appears, select a directory and specify the name of the file to export the tracking results to.
3. After clicking `OK`, the [](../user-interface.md#importexport-tracking-options) dialog will open.
4. Ensure that the `All` option under the `Volumes` section is selected.
5. Press the `OK` button to export the tracking results.

![Export Tracking Results](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_AllVolumes.png)

## Switching to the Tracking Evaluation Module

To access the `Tracking Evaluation module`, follow these steps:
1. Open the module drop-down menu.
2. Navigate to the `Tracking` category.
3. Select the `Tracking Evaluation` module.

![Tracking Evaluation Module](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_SwitchModule.png)

## Loading in Results

To load your tracking results into the `Tracking Evaluation` module, perform the following:
1. In the `Input Data` section, select the file you exported from Autoscoper.
2. Choose the sample data type you wish to compare your results to.
3. Press the `Load Data for Evaluation` button.

```{tip}
You may need to adjust the camera of the 3D scene to visualize the results more clearly.
```

![Load data](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_LoadData.png)

## Visualizing Results

In the `Visualize Results` section, you can:
* Click the `Play` button to automatically scrub through the sequence.
* Use the timeline to manually scrub through the sequence.
* Visualize results with a 1mm or 2 degree difference from the ground truth displayed in green.
* Identify results exceeding the thresholds displayed in red, along with a partially transparent version of the ground truth.

![Results Gif](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/eval_ShowModule.gif)
