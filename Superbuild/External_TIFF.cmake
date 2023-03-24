
set(proj TIFF)

set(${proj}_DEPENDENCIES "")

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

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

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY https://github.com/BrownBiomechanics/libtiff.git
    GIT_TAG a74a58233f8fcd89f52e00678d706246dd365611 # autoscoper-v4.4.0-2022-05-20-b6a17e56
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
      # Options
      -DBUILD_SHARED_LIBS:BOOL=ON
      -Dcxx:BOOL=OFF
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
  elseif(APPLE)
    set(TIFF_LIBRARY ${tiff_DIR}/lib/libtiff.dylib)
  else()
    set(TIFF_LIBRARY ${tiff_DIR}/lib/libtiff.so)
  endif()

  mark_as_superbuild(
    VARS
      TIFF_INCLUDE_DIR:PATH
      TIFF_LIBRARY:PATH
    )

  ExternalProject_Message(${proj} "TIFF_INCLUDE_DIR:${TIFF_INCLUDE_DIR}")
  ExternalProject_Message(${proj} "TIFF_LIBRARY:${TIFF_LIBRARY}")
endif()
