# Camera Calibration File Format

```{note}
Camera Calibration files described below are generated outside of SlicerAutoscoperM. Users are recommended to use [XMALAB](https://bitbucket.org/xromm/xmalab/wiki/Home)
```

## MayaCam 2.0

### Overview

The camera calibration file is a txt and contains all of the information to load a camera into Autoscoper. The file is organized with key value pairs.

### Key value pairs

| Key | Value |
| --- | --- |
| `image size` | The height and width of the radiograph images. |
| `camera matrix` | The intrinsic camera matrix. |
| `rotation` | The rotation matrix. |
| `translation` | The translation matrix. |

### Template

```{note}
The rotation is applied to the camera after the translation. Therefore, the final camera position is calculated by the following equation:

```

```{math}
camera\_position = rotation * -translation
```

```
image size
height,width

camera matrix
fx,0,cx
0,fy,cy
0,0,1

rotation
r00,r01,r02
-r10,-r11,-r12
-r20,-r21,-r22

translation
-x
y
-z
```

### Example

```
image size
1024,1024

camera matrix
1.0,0.0,512.0
0.0,1.0,512.0
0.0,0.0,1.0

rotation
1.0,0.0,-1.0
0.0,1.0,0.0
0.0,0.0,1.0

translation
0.0
0.0
512.0
```

## MayaCam 1.0

### Overview

The camera calibration file is a csv and contains all of the information to load a camera into Autoscoper. The file is organized by line.

### Line values

| Line Number | Description |
| --- | --- |
| 1 | The camera location in world space. |
| 2 | Rotations around the local x, y, and z axes of the camera. The order of rotation is z, y, x (Roll, Pitch, Yaw).|
| 3 | The position of the film plane relative to the camera. Given in the camera's local space. |
| 4 | u0, v0, z |
| 5 | scale, height, width. Scale is ignored and is computed from distance and z. |

### Example

```{code-block} text
:linenos:
-737.5,-514.9,94.3
89.7,-2.0,-55.2
3.9,4.1,-632.5
840.2,839.2,-6325.8
0.1,1760,1760
```

## Backend Calculations

### Overview

This section describes the calculations that are performed by the backend to convert the camera calibration file into an internal Camera object.

### Image Plane


The image plane is the location of the DRR image in world space. The center of the image plane is calculated by the following equation:

```{note}
The below formula applies to VTKCam 1.0 and MayaCam 2.0. For MayaCam 1.0, `c_x`` and `c_y` are replaced by `u0` and `v0` respectively. The value for `z` is also not calculated and is instead pulled from the camera calibration file.

`DRR_X` and `DRR_Y` are the width and height of the DRR image respectively.

The translation matrix is represented by `t` and the rotation matrix is represented by `R`.
```

The constant `-0.5` is used so that the `z` value is the average of the `f_x` and `f_y` values, and it is negated to be consistent with the MayaCam 1.0 file format.

```{math}
z = -0.5 * (f_x + f_y)
```

```{math}
distance = \sqrt{t_x^2 + t_y^2 + t_z^2}
```

The constant `-1.5` is used so that the image plane is placed across the origin from the camera.

```{math}
scale = \frac{-1.5 * distance}{z}
```

```{math}
image\_plane\_translation[0] = scale *  (\frac{DRR_X}{2} - c_x)
```

```{math}
image\_plane\_translation[1] = scale *  (\frac{DRR_Y}{2}  - c_y)
```

```{math}
image\_plane\_translation[2] = scale *  z
```

```{math}
image\_plane\_center = R * image\_plane\_translation + t
```

### Viewport

The viewport defines the position of the 2D viewer. The viewport is calculated by the following equation:

```{note}
The below formula applies to VTKCam 1.0 and MayaCam 2.0. For MayaCam 1.0, c_x and c_y are replaced by u0 and v0 respectively. While f_x and f_y are replaced by z.

DRR_X and DRR_Y are the width and height of the DRR image respectively.
```

```{math}
viewport[0] =  -(2 * c_x - DRR_X) / f_x
```

```{math}
viewport[1] =  -(2 * c_y - DRR_Y) / f_y
```

```{math}
viewport[2] =  2 * DRR_X / f_x
```

```{math}
viewport[3] =  2 * DRR_Y / f_y
```

### VTKCam 1.0 Unique Calculations

The VTKCam 1.0 file format calculates the camera orientation and the focal length from the camera position, focal point, and view angle.

#### Camera Orientation

```{note}
The translation matrix is assumed to be the camera position.
```

First, a vector is calculated from the camera position to the focal point.

```{math}
f = \text{focal_point} - \text{camera_position}

\hat{f} = \frac{f}{\lVert f \rVert}
```

Next, a side vector is calculated from the cross product of the focal vector and the up vector. The up vector is assumed to be (0, 1, 0).

```{math}
s = \hat{f} \times \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}

\hat{s} = \frac{s}{\lVert s \rVert}
```

Finally, the up vector is calculated from the cross product of the side vector and the focal vector.

```{math}
\text{up} = \hat{s} \times \hat{f}
```

The rotation matrix is then calculated from the side, up, and focal vectors.

```{math}
\text{rotation_matrix} = \begin{bmatrix}
s_x & s_y & s_z \\
up_x & up_y & up_z \\
-f_X & -f_y & -f_z \\
\end{bmatrix}
```

#### Focal Length

```{note}
The principal point (c_x, c_y) is assumed to be half of the image size in the x and y directions respectively.

The view angle is assumed to be in radians (This is converted from degrees to radians in the backend.).
```

The focal length is calculated from the view angle and the image size.

```{math}
f_x = \frac{image\_size_x}{2 \cdot \tan(\frac{view\_angle}{2})}
```

```{math}
f_y = \frac{image\_size_y}{2 \cdot \tan(\frac{view\_angle}{2})}
```
## VTKCam 1.0

```{important} Deprecated
This file specification is no longer supported. The VRG generation functionality was removed in [BrownBiomechanics/SlicerAutoscoperM#140](https://github.com/BrownBiomechanics/SlicerAutoscoperM/pull/140). Please refer to the updated workflows and documentation for alternative solutions.
```

### Overview

The camera calibration file is a yaml and contains all of the information to load a camera into Autoscoper. The file is organized with key value pairs.

:::{warning}
The VTKCam 1.0 format is non-standard and exists purely as an
implementation detail. While we are documenting it, the file
organization may change from version to version without notice.
We mean it.
:::

### Key value pairs

| Key | Value |
| --- | --- |
|`version`| The version of the file. Currently only 1.0 is supported. |
|`focal-point`| The XYZ coordinates of the focal point. |
|`camera-position`| The XYZ coordinates of the camera position. |
|`view-up`| The up vector of the camera. |
|`view-angle`| The view angle of the camera. |
|`image-width`| The width of the image. |
|`image-height`| The height of the image. |
|`clipping-range`| The range of the closest and farthest objects that will affect the rendering. |

::: {warning}
The `clipping-range` is not currently used by Autoscoper. This is used to
communicate information within the AutoscoperM Slicer extension.
:::

### Example

```{code-block} json
{
  "@schema": "https://autoscoperm.slicer.org/vtk-schema-1.0.json",
  "version": 1.0,
  "focal-point": [-7.9999999999999964, -245.50000000000006, -186.65000000000006],
  "camera-position": [104.71926635196253, -255.22259800818924, -179.66771669788898],
  "view-up": [0.0, 1.0, 0.0],
  "view-angle": 30.0,
  "image-width": 1760,
  "image-height": 1760,
  "clipping-range": [0.1, 1000],
}
```
