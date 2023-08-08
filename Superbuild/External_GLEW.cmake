
set(proj GLEW)

set(${proj}_DEPENDENCIES "")

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

  if(UNIX AND NOT APPLE)
    if(NOT DEFINED OpenGL_GL_PREFERENCE)
      set(OpenGL_GL_PREFERENCE "LEGACY")
    endif()
    if(NOT "${OpenGL_GL_PREFERENCE}" MATCHES "^(LEGACY|GLVND)$")
      message(FATAL_ERROR "OpenGL_GL_PREFERENCE variable is expected to be set to LEGACY or GLVND")
    endif()
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DOpenGL_GL_PREFERENCE:STRING=${OpenGL_GL_PREFERENCE}
      )
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY https://github.com/BrownBiomechanics/glew.git
    GIT_TAG 8248234b584a17528a499610c9dec13db8c9acba # autoscoper-2.2.0-2020-03-15-9fb23c3e6
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
      # Options
      -DBUILD_UTILS:BOOL=OFF
      # Install directories
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_INSTALL_BINDIR:STRING=${Autoscoper_BIN_DIR}
      -DCMAKE_INSTALL_LIBDIR:STRING=${Autoscoper_LIB_DIR}
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )
  set(GLEW_DIR ${EP_INSTALL_DIR}/${Autoscoper_LIB_DIR}/cmake/glew)
  ExternalProject_Message(${proj} "GLEW_DIR:${GLEW_DIR}")
  mark_as_superbuild(GLEW_DIR:PATH)
endif()
