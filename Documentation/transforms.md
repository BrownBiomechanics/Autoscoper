# Transforms

## Background

:::{figure-md} Slicer-coordinate-systems
:align: center

![Coordinate systems](https://github.com/Slicer/Slicer/releases/download/docs-resources/coordinate_systems.png)

World coordinate system (left), Anatomical coordinate system (middle), Image coordinate system (right) Image source: [Slicer Coordinate Systems documentation](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html)
:::

In 3D Slicer, medical image data is processed using the Right-Anterior-Superior (RAS) coordinate
system by default. However, to maintain compatibility with most medical imaging software, which
typically uses the Left-Posterior-Superior (LPS) coordinate system, Slicer
assumes files are stored in the LPS coordinate system unless otherwise specified.

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_XYZ_axes.png
:align: center
:alt: XYZ Axes
```

When reading or writing files, Slicer automatically converts between RAS and LPS
coordinate systems as needed. This conversion involves flipping the sign of the first
two coordinate axes. The transformation matrix for converting between RAS and LPS
coordinates is shown below:

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_LPS_to_RAS_coords.png
:align: center
:alt: LPS to RAS Coordinate Transformation
```

For more details on the coordinate systems used in 3D Slicer, refer to the [Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html).

### Spatial referencing

Spatial referencing defines the relationship between voxel coordinates in image space
and their corresponding positions in world space. This includes encoding voxel-to-world
unit resolution (also known as pixel/voxel spacing), the origin, and orientation of the
dataset.

In the DICOM format, spatial referencing information is embedded in the DICOM header
metadata. In MATLAB, the [`imref3d`](https://www.mathworks.com/help/images/ref/imref3d.html) function can be constructed
using this metadata to store the intrinsic spatial relationships.

Transforming data between image space and world space involves proportional adjustments
to the image volume, as shown in the matrix below:

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Image_to_world_space.png
:align: center
:alt: Image space to World Space
```

In the matrix shown above, `P_{x,y,z}` represents the pixel spacing along each axis, while `O_{x,y,z}` defines the origin for each dimension.

When importing DICOM data into 3D Slicer, a spatial referencing transform is automatically applied to the CT image volume.
This transform is derived from the DICOM header metadata and aligns the data to Slicer’s RAS anatomical orientation system,
with the +X, +Y, and +Z directions corresponding to the red, green, and blue axes, respectively.

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Slicer_DICOM_data.png
:align: center
:alt: DICOM Data Loaded into Slicer
```

The spatial referencing information of a volume in Slicer can be inspected in the **Volume Information** section of the **Volumes** module. This provides detailed metadata, including spacing, origin, and orientation.

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Volume_spatial_info.png
:align: center
:alt: Spatial Referencing Information of a Volume
```

## Transforms in SlicerAutoscoperM

SlicerAutoscoperM facilitates the tracking of rigid bodies in world space. When tracking multiple
rigid bodies, their relative motions can be calculated with respect to a designated reference body.

### Pre-Processing and Volume Generation

In the Pre-processing tab of the SlicerAutoscoperM module, rigid bodies of interest can
be automatically segmented from a CT DICOM volume. Alternatively, previously generated segmentations
can be loaded. These segmentations are used to generate partial volumes, isolating the density data
within the bounds of the segmented model. The partial volume is saved as a TIFF stack, which is later imported into Autoscoper in a specific orientation referred to as Autoscoper (AUT) space.

Within Autoscoper, this TIFF stack is used to generate Digitally Reconstructed Radiographs (DRRs)
by projecting the density data onto the target image plane, where it is overlaid with the radiographs for each frame.

TIFF stacks, like CT DICOM volumes, originate in image space and require a
spatial referencing transform to encode their world location, spacing, and
orientation. In Autoscoper, the voxel resolution of TIFF volumes aligns with predefined axes.

```{image} https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Model_and_its_TIFF_pv.png
:align: center
:alt: Segmented Model and Corresponding TIFF Stack
```

### Output Directory Structure

During partial volume generation, SlicerAutoscoperM creates several folders in the output directory. These folders store the models, volumes, transforms, and tracking information required for processing. The structure is as follows:

```
Output Directory
│
├── Models
│   └── AUT{bone}.stl
│
├── Tracking
│   └── {bone}.tra
│
├── Transforms
│   ├── {bone}.tfm
│   ├── {bone}_DICOM2AUT.tfm
│   ├── {bone}_PVOL2AUT.tfm
│   ├── {bone}_scale.tfm
│   └── {bone}_t.tfm
│
└── Volumes
    └── {bone}.tif
```

**Folder Descriptions:**
* **Models:**
  * `AUT{bone}.stl`: Mesh file of the volume segmentation placed in Autoscoper (AUT) space.
* **Tracking:**
  * `{bone}.tra`: Equivalent to `{bone}_DICOM2AUT.tfm` but formatted for Autoscoper compatibility.
* **Transforms:**
  * `{bone}.tfm`: Non-rigid transform that translates and scales the `{bone}.tif` volume to its spatial location within the segmented CT-DICOM.
  * `{bone}_DICOM2AUT.tfm`: Transformation from DICOM space into Autoscoper (AUT) space.
  * `{bone}_PVOL2AUT.tfm`: Transformation from world space into Autoscoper (AUT) space.
  * `{bone}_scale.tfm`: Scaling matrix that converts the volume from image space to world space.
  * `{bone}_t.tfm`: Translation matrix moving between the world origin and the location of the partial volume within the segmented CT-DICOM.
* **Volumes:**
  * `{bone}.tif`: Non-spatially transformed volumetric data segmented from CT-DICOM.


### Visualizing Transformations

The following diagrams illustrate the transformations between spaces:
:::{figure-md} Tfms-to-dicom-space
:align: center

![Transforms of TIFF Stack to DICOM Space](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Tfms_to_DICOM_space.png)

Transforms to DICOM space: `{bone}.tfm` (pink arrow), `{bone}_t.tfm` (blue arrow)
:::


:::{figure-md} Tfms-to-aut-space
:align: center

![Transforms of PV to AUT Space](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/transforms_Tfms_to_AUT_space.png)

Transforms to Autoscoper space: `{bone}_DICOM2AUT.tfm` (orange arrow), `{bone}_t.tfm` (blue arrow) and `{bone}_PVOL2AUT.tfm` (gray arrow)
:::

### Transformation Workflow

The relationships between image, world, and Autoscoper spaces are illustrated below:
:::{mermaid}
:align: center

flowchart TD
    subgraph image_space["Image Space"]
      tiff["Bone TIFF"]
    end
    subgraph world_space["World Space"]
      partial_volume["Bone Partial Volume (PVOL)"]
      transformed_pvol["Spatially Located PVOL"]
      world_origin(["World origin"])
    end
    subgraph aut_space["Autoscoper Space (AUT)"]
      model["Bone Model"]
      aut_transformed_pvol["AUT-Space Located PVOL"]
    end
    image_space -- "{bone}.tfm" --> transformed_pvol
    image_space -- "{bone}_scale.tfm" --> partial_volume
    image_space -- "{bone}_DICOM2AUT.tfm" --> aut_transformed_pvol
    transformed_pvol -- "{bone}_PVOL2AUT.tfm" --> aut_transformed_pvol
    partial_volume -- "{bone}_t.tfm" --> transformed_pvol
:::
