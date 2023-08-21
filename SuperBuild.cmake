
set(Autoscoper_DEPENDENCIES
  GLEW
  TIFF
  )
if(Autoscoper_RENDERING_BACKEND STREQUAL "OpenCL")
  if(Autoscoper_OPENCL_USE_ICD_LOADER)
    list(APPEND Autoscoper_DEPENDENCIES
      OpenCL-ICD-Loader
      )
  endif()
endif()

if(Autoscoper_BUILD_VTK)
  list(APPEND Autoscoper_DEPENDENCIES
    VTK
   )
endif()

if(Autoscoper_BUILD_VTK)
  list(APPEND Autoscoper_DEPENDENCIES
    VTK
   )
endif()

set(proj ${SUPERBUILD_TOPLEVEL_PROJECT})

ExternalProject_Include_Dependencies(${proj}
  PROJECT_VAR proj
  SUPERBUILD_VAR Autoscoper_SUPERBUILD
  DEPENDS_VAR Autoscoper_DEPENDENCIES
  )

ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/${Autoscoper_BINARY_INNER_SUBDIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    # Compiler settings
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DQt5_DIR:PATH=${Qt5_DIR}
    -DVTK_DIR:PATH=${VTK_DIR}
    # Options
    -DAutoscoper_SUPERBUILD:BOOL=OFF
    -DAutoscoper_SUPERBUILD_DIR:PATH=${CMAKE_BINARY_DIR}
    # Dependencies
    -DAutoscoper_DEPENDENCIES:STRING=${Autoscoper_DEPENDENCIES}
  DEPENDS
    ${Autoscoper_DEPENDENCIES}
  INSTALL_COMMAND ""
)

ExternalProject_AlwaysConfigure(${proj})
