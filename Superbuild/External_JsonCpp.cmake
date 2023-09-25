
set(proj JsonCpp)

set(${proj}_DEPENDENCIES "")

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(Autoscoper_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling Autoscoper_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED JsonCpp_INCLUDE_DIR AND NOT EXISTS ${JsonCpp_INCLUDE_DIR})
  message(FATAL_ERROR "JsonCpp_INCLUDE_DIR variable is defined but corresponds to nonexistent directory")
endif()
if(DEFINED JsonCpp_LIBRARY AND NOT EXISTS ${JsonCpp_LIBRARY})
  message(FATAL_ERROR "JsonCpp_LIBRARY variable is defined but corresponds to nonexistent file")
endif()

if((NOT DEFINED JsonCpp_INCLUDE_DIR
   OR NOT DEFINED JsonCpp_LIBRARY) AND NOT Autoscoper_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
    GIT_TAG 5defb4ed1a4293b8e2bf641e16b156fb9de498cc # 1.9.5
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
      -DJSONCPP_WITH_TESTS:BOOL=OFF
      -DJSONCPP_WITH_POST_BUILD_UNITTEST:BOOL=OFF
      -DJSONCPP_WITH_WARNING_AS_ERROR:BOOL=OFF
      -DJSONCPP_WITH_PKGCONFIG_SUPPORT:BOOL=OFF
      -DJSONCPP_WITH_CMAKE_PACKAGE:BOOL=ON
      -DBUILD_SHARED_LIBS:BOOL=OFF
      -DBUILD_STATIC_LIBS:BOOL=ON
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_BINDIR:STRING=${Autoscoper_BIN_DIR}
      -DCMAKE_INSTALL_LIBDIR:STRING=${Autoscoper_LIB_DIR}
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(${proj}_INCLUDE_DIR ${EP_INSTALL_DIR}/include)
  set(${proj}_LIBRARY ${EP_INSTALL_DIR}/${Autoscoper_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}jsoncpp${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()

# For sake of consistency with how Slicer integrates JsonCpp, we ignore the "jsoncppConfig.cmake" file provided by jsoncpp and
# instead set JsonCpp_INCLUDE_DIR and JsonCpp_LIBRARY expected by the "FindJsonCpp" CMake module.
mark_as_superbuild(
  VARS
    ${proj}_INCLUDE_DIR:PATH
    ${proj}_LIBRARY:FILEPATH
)

ExternalProject_Message(${proj} "${proj}_INCLUDE_DIR:${${proj}_INCLUDE_DIR}")
ExternalProject_Message(${proj} "${proj}_LIBRARY:${${proj}_LIBRARY}")
