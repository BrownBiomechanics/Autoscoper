# Transforms

## Background

| ![Coordinate systems](https://github.com/Slicer/Slicer/releases/download/docs-resources/coordinate_systems.png) |
| -- |
| <p style="text-align: center;"> World coordinate system (left), Anatomical coordinate system (middle), Image coordinate system (right) Image source: <a href="https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html">Slicer Coordinate Systems documentation</a></p>|


In 3D Slicer, medical image data is processed using the Right-Anterior-Superior (RAS) coordinate
system by default. However, to maintain compatibility with other medical imaging software, Slicer
assumes that files are stored in the Left-Posterior-Superior (LPS) coordinate system unless
otherwise specified. When reading or writing files, Slicer may need to flip the sign of the first
two coordinate axes to convert between RAS and LPS.
For more details on the coordinate systems used in 3D Slicer, refer to the [Slicer documentation](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html).

The following transformation matrix converts between RAS and LPS coordinates:

![XYZ Axis](https://)

```{math}
\begin{bmatrix}
-1 & 0 & 0 & 0\\
0 & -1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
```
### Spatial referencing

Spatial referencing data is used to encode voxel to world unit resolution (also referred to as pixel/voxel spacing), origin,
and orientation. In DICOM format, the spatial referencing can be retrieved from the DICOM header meta data. In MATLAB,
the `imref3d` function can be constructed from the DICOM meta data to store the intrinsic spatial relationship.

Transforming between image and world spaces has a visual proportion change to the image volume:

![Image space to World Space](https://)

The matrix below can be used to convert from image space to world space, where `P_{x,y,z}` represents pixel spacing along
each axis, and `O_{x,y,z}` is the origin for each respective dimension:

```{math}
\begin{bmatrix}
P_x & 0 & 0 & O_x\\
0 & P_y & 0 & O_y\\
0 & 0 & P_z & O_z\\
0 & 0 & 0 & 1
\end{bmatrix}
```

When importing DICOM data into Slicer, a spatial referencing transform is automatically applied to a CT image volume based on the
header metadata in the DICOM header. Note the RAS anatomical orientation and +X, +Y, +Z (red/green/blue) axes indicators.

<!-- Lots of images/diagrams to bring over -->

## Transforms in SlicerAutoscoperM

SlicerAutoscoperM enables tracking of rigid bodies in the 'World' space. When tracking multiple
rigid bodies, their relative motions can be computed with respect to a reference body.

In the AutoscoperM Slicer module, under the Pre-processing tab, rigid bodies of interest can
be automatically segmented from a CT DICOM volume. Alternatively, previously generated segmentations
can be loaded. These segmentations are used to generate partial volumes, where the density data
within the bounds of the segmented model is isolated and saved as a TIFF stack. In Autoscoper, this
TIFF stack is imported in a specific orientation, referred to as Autoscoper (AUT) space. The data from the
partial volume is projected onto the target image plane (overlaid with the radiograph for each frame)
as a Digitally Reconstructed Radiograph (DRR).

Like CT DICOM volumes, TIFF stacks are initially in image space and require a
spatial referencing transform to describe their world location, spacing, and
orientation. In Autoscoper, the voxel resolution of TIFF volumes follows a specified axes.

To facilitate post-processing of output transforms from Autoscoper, several folders
are populated at the time of Partial Volume Generation in SlicerAutoscoperM. The
following structure outlines the information in the output directory:

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

* **Models:**
  * `AUT{bone}.stl`: Mesh file of the volume segmentation placed in Autscoper (AUT) space.
* **Tracking:**
  * `{bone}.tra`: Equivalent to `{bone}_DICOM2AUT.tfm` but formatted for Autoscoper compatibility.
* **Transforms:**
  * `{bone}.tfm`: Non-rigid transform that translates and scales the `{bone}.tif` volume to its spatial location within the segmented CT-DICOM.
  * `{bone}_DICOM2AUT.tfm`: Transformation from DICOM space into Autoscoper (AUT) space.
  * `{bone}_PVOL2AUT.tfm`: Transformation from world space into Autoscoper (AUT) space.
  * `{bone}_scale.tfm`: Scaling matrix that converting the volume from image space to world space.
  * `{bone}_t.tfm`: Translation matrix moving between the world origin and the location of the partial volume within the segmented CT-DICOM.
* **Volumes:**
  * `{bone}.tif`: Non-spatially transformed volumetric data segmented from CT-DICOM.


```{mermaid}
flowchart TD
    subgraph image_space["Image Space"]
      tiff["TIFF"]
    end
    subgraph dicom_space["DICOM Space"]
      transformed_pvol["Spatially located PV"]
    end
    subgraph world_space["World Space"]
      world_origin["World origin"]
      partial_volume["Partial Volume (PVOL)"]
    end
    subgraph aut_space["Autoscoper Space (AUT)"]
      model["Model"]
    end
    world_space -- "{bone}_t.tfm" --> dicom_space
    image_space -- "{bone}.tfm" --> dicom_space
    image_space -- "{bone}_scale.tfm" --> world_space
    image_space -- "{bone}_DICOM2AUT.tfm" --> aut_space
    world_space -- "{bone}_PVOL2AUT.tfm" --> aut_space
```
