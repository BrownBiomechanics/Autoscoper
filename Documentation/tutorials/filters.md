# Rendering and Filters

Filters are used to modify the rendering of the DRRs and Radiographs. This is an important step, as the quality of the filters will impact the quality of the registration. The filters are applied to the DRRs and Radiographs before the registration is performed. 

For a brief overview of the rendering options UI, see that section of the [UI Overview](../user-interface.md#rendering-options).

```{note}
Enhancing the image features using filters is a necessary step to achieve an accurate match. Define the parameters of the four filters, including contrast (intensity detection), Sobel (edge detection), Gaussian (blurring/smoother), and Sharpen (boldening the edges) in the software. These filters can be selected by right-clicking on the Rad Renderer or DRR Renderer objects shown in the top-left widget.
```

## Types of Filters

There are four types of filters that can be applied to the DRRs and Radiographs. These are:

1. Sobel Filter
2. Contrast Filter
3. Gaussian Filter
4. Sharpen Filter

## Adding a new Filter

To add a new filter, hover your mouse over the renderer that you want to add the filter to. Then, `right click` and select hover over the `Add Filter` option. This will display a list of the filters that can be added to the renderer. Select the filter that you want to add.

![Add Filter](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_FilterAdd.png)

## Removing a Filter

To remove a filter, hover your mouse over the filter that you want to remove. Then, `right click` and select the `Remove Filter` option.

![Remove Filter](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_FilterRemove.png)

## Adjusting Filter Parameters

To adjust the parameters of a filter, click on the wrench icon next to the filter. This will open a drop down menu with the parameters that can be adjusted.

![Adjust Filter Parameters](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_FilterAdjust.png)

### Sobel Filter

The Sobel filter is used to detect edges in the DRRs and Radiographs. This filter is useful for detecting the edges of bones in the images. The Sobel filter has two parameters that can be adjusted. These are:

1. `Blend` - This parameter controls the amount of blending that is applied to the filter. The default value is 0.5.
2. `Scale` - This parameter controls the scale of the filter. The default value is 1.0.

### Contrast Filter

The Contrast filter is used to adjust the contrast of the DRRs and Radiographs. This filter is useful for adjusting the contrast of the images to make the edges more visible. The Contrast filter has two parameters that can be adjusted. These are:

1. `Alpha` - This parameter controls the alpha value of the filter. The default value is 1.0.
2. `Beta` - This parameter controls the beta value of the filter. The default value is 1.0.

### Gaussian Filter

The Gaussian filter is used to blur the DRRs and Radiographs. This filter is useful for removing noise from the images. The Gaussian filter has one parameter that can be adjusted. This is:

1. `Radius` - This parameter controls the radius of the filter. The default value is 1.0.

### Sharpen Filter

The Sharpen filter is used to sharpen the DRRs and Radiographs. This filter is useful for making the edges of the images more visible. The Sharpen filter has two parameters that can be adjusted. These are:

1. `Radius` - This parameter controls the radius of the filter. The default value is 1.0.
2. `Contrast` - This parameter controls the contrast of the filter. The default value is 1.0.

### DRR Renderer Settings

The DRR Renderer has three settings that can be adjusted. These are:

1. `Sample Distance` - This parameter controls the sample distance of the ray casting. The default value is 0.62.
2. `XRay Intensity` - This parameter controls the intensity of the XRay. The default value is 0.5.
3. `XRay Cutoff` - This parameter controls the cutoff of the XRay. The default value is 0.0.

## Saving and Loading Filters

Filters can be saved and loaded for each camera in the scene. To save a filter, hover your mouse over the camera that you want to save the filter for. Then, `right click` and select the `Save Filters` option. This will open a dialog box where you can enter the name of the filter. To load a filter, hover your mouse over the camera that you want to load the filter for. Then, `right click` and select the `Load Filters` option. This will open a dialog box where you can select the filter that you want to load.

![Save and Load Filters](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/tutorial_FilterSaveLoad.png)