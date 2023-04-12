# Building AutoscoperM

## Building Slicer
Being an extension of 3D Slicer, the first step is to build Slicer from source. The instructions for building Slicer can be found [here](https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html).

## Building AutoscoperM
Once Slicer is built, the next step is to build AutoscoperM.

* Clone the AutoscoperM respotory into its own directory: 
    * `git clone https://github.com/BrownBiomechanics/SlicerAutoscoperM.git`
* Create a build directory for AutoscoperM (it is recomened to do an out of source build):
    * `mkdir SlicerAutoscoperM-build`
* The build configuration has the following options:
    * `Slicer_DIR`: The path to the Slicer build directory.
    * `Qt5_DIR`: The path to the Qt5Config.cmake file.
    * `SlicerAutoscoperM_SUPERBUILD`: If checked, the external dependencies will be built.
        * This is required to build Autoscoper as apart of the extension.
* Configure the project with CMake:
    * `cmake ../path/to/source/ -DSlicer_DIR=/path/to/Slicer-build/ -DQt5_DIR=/path/to/Qt5Config.cmake -DSlicerAutoscoperM_SUPERBUILD=ON`
* Select a configuration type and build the project:
    * `cmake --build . --config Release`
* Once the project is built, the extension can be loaded by launching the `SlicerWithSlicerAutoscoperM` executable located inside the `inner-build` folder in the build directory.
