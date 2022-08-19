
set(proj TIFF)

set(${proj}_DEPENDENCIES "")

if(Autoscoper_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling Autoscoper_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED TIFF_INCLUDE_DIR AND NOT EXISTS ${TIFF_INCLUDE_DIR})
  message(FATAL_ERROR "TIFF_INCLUDE_DIR variable is defined but corresponds to nonexistent directory")
endif()
if(DEFINED TIFF_LIBRARY AND NOT EXISTS ${TIFF_LIBRARY})
  message(FATAL_ERROR "TIFF_LIBRARY variable is defined but corresponds to nonexistent file")
endif()

if((NOT DEFINED TIFF_INCLUDE_DIR
   OR NOT DEFINED TIFF_LIBRARY) AND NOT Autoscoper_USE_SYSTEM_${proj})

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
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG b6a17e567f143fab49734a9e09e5bafeb6f97354
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      # Compiler settings
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(tiff_DIR ${EP_INSTALL_DIR})
  set(TIFF_ROOT ${tiff_DIR})
  set(TIFF_INCLUDE_DIR ${tiff_DIR}/include)
  if(WIN32)
    set(TIFF_LIBRARY $<IF:$<CONFIG:Debug>, ${tiff_DIR}/lib/tiffd.lib, ${tiff_DIR}/lib/tiff.lib>)
  else()
    set(TIFF_LIBRARY ${tiff_DIR}/lib/libtiff.a)
  endif()
  message(STATUS "SuperBuild - TIFF_INCLUDE_DIR: ${TIFF_INCLUDE_DIR}")
  message(STATUS "SuperBuild - TIFF_LIBRARY: ${TIFF_LIBRARY}")
endif()
