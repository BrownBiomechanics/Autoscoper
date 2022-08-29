
set(proj GLEW)

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(Autoscoper_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling Autoscoper_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED GLEW_DIR AND NOT EXISTS ${GLEW_DIR})
  message(FATAL_ERROR "GLEW_DIR variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED GLEW_DIR AND NOT Autoscoper_USE_SYSTEM_${proj})

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
    ${${proj}_EP_ARGS}
    URL https://sourceforge.net/projects/glew/files/glew/2.2.0/glew-2.2.0.zip
    URL_MD5 970535b75b1b69ebd018a0fa05af63d1
    SOURCE_DIR ${EP_SOURCE_DIR}
    SOURCE_SUBDIR build/cmake
    BINARY_DIR ${EP_BINARY_DIR}
    INSTALL_DIR ${EP_INSTALL_DIR}
    CMAKE_CACHE_ARGS
      # Compiler settings
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(GLEW_DIR ${EP_INSTALL_DIR}/lib/cmake/glew)
  ExternalProject_Message(${proj} "GLEW_DIR:${GLEW_DIR}")
  mark_as_superbuild(GLEW_DIR:PATH)
endif()
