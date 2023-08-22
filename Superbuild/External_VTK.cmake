
set(proj VTK)

set(${proj}_DEPENDENCIES "")

if(Autoscoper_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling Autoscoper_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED VTK_INCLUDE_DIR AND NOT EXISTS ${VTK_INCLUDE_DIR})
  message(FATAL_ERROR "VTK_INCLUDE_DIR variable is defined but corresponds to nonexistent directory")
endif()
if(DEFINED VTK_LIBRARY AND NOT EXISTS ${VTK_LIBRARY})
  message(FATAL_ERROR "VTK_LIBRARY variable is defined but corresponds to nonexistent file")
endif()

if((NOT DEFINED VTK_INCLUDE_DIR
   OR NOT DEFINED VTK_LIBRARY) AND NOT Autoscoper_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  if(NOT CMAKE_CONFIGURATION_TYPES)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      )
  endif()

  ExternalProject_Add(${proj}
    GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk.git
    GIT_TAG f2c452c9c42005672a3f3ed9218dd9a7fecca79a # v9.2.6
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      # Compiler settings
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      -DVTK_USE_CUDA:BOOL=${Autoscoper_BUILD_WITH_CUDA}
      # Options
      -DBUILD_SHARED_LIBS:BOOL=ON
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
    set(VTK_DIR ${EP_BINARY_DIR})
    set(VTK_SOURCE_DIR ${EP_SOURCE_DIR})
    ExternalProject_Message(${proj} "VTK_DIR:${VTK_DIR}")
    ExternalProject_Message(${proj} "VTK_SOURCE_DIR:${VTK_SOURCE_DIR}")
    mark_as_superbuild(VTK_SOURCE_DIR:PATH)
    mark_as_superbuild(
        VARS VTK_DIR:PATH
        LABELS "FIND_PACKAGE"
    )
endif()