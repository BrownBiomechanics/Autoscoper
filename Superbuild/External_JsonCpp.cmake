
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
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DBUILD_STATIC_LIBS:BOOL=OFF
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_BINDIR:STRING=${Autoscoper_BIN_DIR}
      -DCMAKE_INSTALL_LIBDIR:STRING=${Autoscoper_LIB_DIR}
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(JSONCPP_DIR ${EP_INSTALL_DIR})
  set(JSONCPP_SOURCE_DIR ${EP_SOURCE_DIR})
  set(JSONCPP_INCLUDE_DIR ${JSONCPP_DIR}/include)
  set(JSONCPP_LIBRARY ${JSONCPP_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}jsoncpp${CMAKE_SHARED_LIBRARY_SUFFIX})

  mark_as_superbuild(JSONCPP_DIR JSONCPP_SOURCE_DIR JSONCPP_INCLUDE_DIR JSONCPP_LIBRARY)
endif()
