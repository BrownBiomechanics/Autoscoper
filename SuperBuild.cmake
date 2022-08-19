include(ExternalProject)

set(Autoscoper_DEPENDENCIES
  GLEW
  TIFF
  )

foreach(dependency IN LISTS Autoscoper_DEPENDENCIES)
  message(STATUS "SuperBuild - Adding ${dependency}")
  include(${CMAKE_CURRENT_SOURCE_DIR}/Superbuild/External_${dependency}.cmake)
endforeach()

ExternalProject_Add(Autoscoper
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/${Autoscoper_BINARY_INNER_SUBDIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CMAKE_CACHE_ARGS
    # Compiler settings
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    # Options
    -DAutoscoper_SUPERBUILD:BOOL=OFF
    -DAutoscoper_BUILD_WITH_CUDA:BOOL=${Autoscoper_BUILD_WITH_CUDA}
    # Dependencies
    -DGLEW_DIR:PATH=${GLEW_DIR}
    -DTIFF_LIBRARY:FILEPATH=${TIFF_LIBRARY}
    -DTIFF_INCLUDE_DIR:PATH=${TIFF_INCLUDE_DIR}
  DEPENDS
    ${Autoscoper_DEPENDENCIES}
  INSTALL_COMMAND ""
)
