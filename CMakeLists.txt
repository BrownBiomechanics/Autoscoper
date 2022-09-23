cmake_minimum_required(VERSION 3.8)

project(Autoscoper)

#-----------------------------------------------------------------------------
# Setting C++ Standard
#-----------------------------------------------------------------------------
set(_msg "Setting C++ standard")
message(STATUS "${_msg}")
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "${_msg} - C++${CMAKE_CXX_STANDARD}")
if(NOT CMAKE_CXX_STANDARD MATCHES "^(14|17|20)$")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD must be set to 14, 17 or 20")
endif()

#-----------------------------------------------------------------------------
include("${Autoscoper_SOURCE_DIR}/CMake/PreventInSourceBuilds.cmake")
AssureOutOfSourceBuilds()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

if(NOT DEFINED Autoscoper_BIN_DIR)
  set(Autoscoper_BIN_DIR bin)
endif()
if(NOT DEFINED Autoscoper_LIB_DIR)
  set(Autoscoper_LIB_DIR lib)
endif()

#-----------------------------------------------------------------------------
# Build with Cuda Option - Enabled by default
#-----------------------------------------------------------------------------
option(Autoscoper_BUILD_WITH_CUDA "Build With CUDA instead of OpenCL" ON)

#-----------------------------------------------------------------------------
# Build with VTK Option - Enabled by default
#-----------------------------------------------------------------------------
option(Autoscoper_BUILD_WITH_VTK "Build With VTK" ON)

#-----------------------------------------------------------------------------
# Configure SuperBuild
#-----------------------------------------------------------------------------
set(SUPERBUILD_TOPLEVEL_PROJECT Autoscoper)
include(CMake/ExternalProjectDependency.cmake)
option(Autoscoper_SUPERBUILD "Build ${PROJECT_NAME} and the projects it depends on." ON)
mark_as_advanced(Autoscoper_SUPERBUILD)
set(Autoscoper_BINARY_INNER_SUBDIR Autoscoper-build)

#-----------------------------------------------------------------------------
# Set a default build type if none was specified
#-----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" )
endif()
if(NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Configuring with build type '${CMAKE_BUILD_TYPE}'")
endif()

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(BUILD_SHARED_LIBS ON)

if(Autoscoper_BUILD_WITH_CUDA)
  add_definitions(-DWITH_CUDA)
endif()

#-----------------------------------------------------------------------------
# Dependencies
#-----------------------------------------------------------------------------
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

#-----------------------------------------------------------------------------
# Superbuild script
#-----------------------------------------------------------------------------
if(Autoscoper_SUPERBUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  return()
endif()

add_subdirectory(libautoscoper)
add_subdirectory(autoscoper)

if(WIN32)
  # Autoscoper_DEPENDENCIES and Autoscoper_SUPERBUILD_DIR CMake variables are set
  # in SuperBuild.cmake.
  foreach(dependency IN LISTS Autoscoper_DEPENDENCIES)
    file(GLOB external_dlls "${Autoscoper_SUPERBUILD_DIR}/${dependency}-install/bin/*.dll")
    file(GLOB external_dllsd "${Autoscoper_SUPERBUILD_DIR}/${dependency}-install/bin/*d.dll")
    list(REMOVE_ITEM external_dlls ${external_dllsd})

    install(FILES ${external_dlls} DESTINATION ${Autoscoper_BIN_DIR}/Release CONFIGURATIONS Release COMPONENT Runtime)
    install(FILES ${external_dllsd} DESTINATION ${Autoscoper_BIN_DIR}/Debug CONFIGURATIONS Debug COMPONENT Runtime)
  endforeach()

  #tmp for moving itk install dlls since they don't go into the install folder'
  file(GLOB external_dlls "${Autoscoper_SUPERBUILD_DIR}/ITK-build/bin/Release/*.dll")
  file(GLOB external_dllsd "${Autoscoper_SUPERBUILD_DIR}/ITK-build/bin/Debug/*.dll")
  install(FILES ${external_dlls} DESTINATION ${Autoscoper_BIN_DIR}/Release CONFIGURATIONS Release COMPONENT Runtime)
  install(FILES ${external_dllsd} DESTINATION ${Autoscoper_BIN_DIR}/Debug CONFIGURATIONS Debug COMPONENT Runtime)

  file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/${Autoscoper_BIN_DIR}/Release)
endif()

if(UNIX)
  file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/${Autoscoper_BIN_DIR})
endif()
