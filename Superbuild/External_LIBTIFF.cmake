############
# This code was adapted from https://github.com/Slicer/Slicer/blob/main/SuperBuild/External_zlib.cmake
############
set(proj TIFF)

set(${proj}_DEPENDENCIES "")

if(Autoscoper_USE_SYSTEM_${proj})
  unset(TIFF_ROOT CACHE)
  find_package(TIFF REQUIRED)
  set(TIFF_INCLUDE_DIR ${LIBTIFF_INCLUDE_DIR})
  set(TIFF_LIBRARY ${LIBTIFF_LIBRARIES})
endif()

if(DEFINED TIFF_ROOT AND NOT EXISTS ${TIFF_ROOT})
  message(FATAL_ERROR "TIFF_ROOT variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED TIFF_ROOT AND NOT Autoscoper_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/external)

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
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(tiff_DIR ${EP_INSTALL_DIR})
  set(TIFF_ROOT ${tiff_DIR})
  set(TIFF_INCLUDE_DIR ${tiff_DIR}/include)
  if(WIN32)
    set(TIFF_LIBRARY ${tiff_DIR}/lib/tiff.lib)
  else()
    set(TIFF_LIBRARY ${tiff_DIR}/lib/libtiff.a)
  endif()
endif()
