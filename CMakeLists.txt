project(Autoscoper)

cmake_policy(SET CMP0048 NEW)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

#-----------------------------------------------------------------------------
# Build with Cuda Option - Enabled by default
#-----------------------------------------------------------------------------
option(Autoscoper_BUILD_WITH_CUDA "Build With CUDA instead of OpenCL" ON)

#-----------------------------------------------------------------------------
# Superbuild Option - Enabled by default
#-----------------------------------------------------------------------------
set(SUPERBUILD_TOPLEVEL_PROJECT Autoscoper)
option(Autoscoper_SUPERBUILD "Build ${PROJECT_NAME} and the projects it depends on." ON)
mark_as_advanced(Autoscoper_SUPERBUILD)
set(Autoscoper_BINARY_INNER_SUBDIR Autoscoper-build)

#SET BUILD-TYPE TO RELEASE IF NOT DEFINED
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]")
  add_definitions(-DQT_NO_DEBUG_OUTPUT)
  set(CMAKE_BUILD_RELEASE TRUE)
else()
  set(CMAKE_BUILD_RELEASE FALSE)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(BUILD_SHARED_LIBS ON)

message(STATUS "Building ${PROJECT_NAME} in ${CMAKE_BUILD_TYPE} mode")
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if(Autoscoper_BUILD_WITH_CUDA)
  add_definitions(-DWITH_CUDA)
endif()

#ADD QT4 SUPPORT
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)
#set(QT_USE_QTOPENGL TRUE)
#include(${QT_USE_FILE})
#set(CMAKE_AUTOMOC ON)

#-----------------------------------------------------------------------------
# Superbuild script
#-----------------------------------------------------------------------------
if(Autoscoper_SUPERBUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  return()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/external)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}/external)

add_subdirectory(libautoscoper)
add_subdirectory(autoscoper)

if(WIN32)
  file(GLOB external_dlls "${CMAKE_INSTALL_PREFIX}/external/bin/*.dll")
  file(GLOB external_dllsd "${CMAKE_INSTALL_PREFIX}/external/bin/*d.dll")
  list(REMOVE_ITEM external_dlls ${external_dllsd})

  install(FILES ${external_dlls} DESTINATION bin/Release CONFIGURATIONS Release)
  install(FILES ${external_dllsd} DESTINATION bin/Debug CONFIGURATIONS Debug)

  file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/Release)

endif()

if(UNIX)
  file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
