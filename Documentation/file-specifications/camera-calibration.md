# Camera Calibration File Format

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

```
image size
height,width

camera matrix
fx,0,cx
0,fy,cy
0,0,1

rotation
r00,r01,r02
r10,r11,r12
r20,r21,r22

translation
t0
t1
t2
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
| 2 | Rotations around the local x, y, and z axes of the camera. The order of rotation is z, y, x. |
| 3 | The position of the film plane relative to the camera. Given in the camera's local space. |
| 4 | u0, v0, z |
| 5 | scale, height, width |

### Example

```linenos
-737.5,-514.9,94.3
89.7,-2.0,-55.2
3.9,4.1,-632.5
840.2,839.2,-6325.8
0.1,1760,1760
