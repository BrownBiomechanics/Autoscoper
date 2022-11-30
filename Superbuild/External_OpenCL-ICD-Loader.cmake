
set(proj OpenCL-ICD-Loader)

set(${proj}_DEPENDENCIES
  OpenCL-Headers
  )

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(Autoscoper_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling Autoscoper_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED OpenCLICDLoader_DIR AND NOT EXISTS ${OpenCLICDLoader_DIR})
  message(FATAL_ERROR "OpenCLICDLoader_DIR variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED OpenCLICDLoader_DIR AND NOT Autoscoper_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  if(NOT CMAKE_CONFIGURATION_TYPES)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      )
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG v2022.09.30
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      # Compiler settings
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      # Options
      -DBUILD_TESTING:BOOL=OFF
      -DOPENCL_ICD_LOADER_BUILD_TESTING:BOOL=OFF
      -DENABLE_OPENCL_LAYERS:BOOL=OFF
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      # Depdendencies
      -DOpenCLHeaders_DIR:PATH=${OpenCLHeaders_DIR}
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    INSTALL_COMMAND ""
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(OpenCLICDLoader_DIR ${EP_BINARY_DIR}/OpenCLICDLoader)

endif()

mark_as_superbuild(
  VARS
    OpenCLICDLoader_DIR:PATH
  )

ExternalProject_Message(${proj} "OpenCLICDLoader_DIR:${OpenCLICDLoader_DIR}")
