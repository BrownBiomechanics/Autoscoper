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
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_CACHE_ARGS
    # Compiler settings
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    # Options
    -DAutoscoper_SUPERBUILD:BOOL=OFF
    -DAutoscoper_BUILD_WITH_CUDA:BOOL=${Autoscoper_BUILD_WITH_CUDA}
  DEPENDS
    ${Autoscoper_DEPENDENCIES}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_BINARY_DIR}
)
