cmake_minimum_required(VERSION 3.8)

project(Autoscoper)

#-----------------------------------------------------------------------------
# Setting C++ Standard
#-----------------------------------------------------------------------------
set(_msg "Setting C++ standard")
message(STATUS "${_msg}")
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "${_msg} - C++${CMAKE_CXX_STANDARD}")
if(NOT CMAKE_CXX_STANDARD MATCHES "^(17|20)$")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD must be set to 17 or 20")
endif()

#-----------------------------------------------------------------------------
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

set(EXTERNAL_PROJECT_DIR "${CMAKE_SOURCE_DIR}/Superbuild")
set(SUPERBUILD_TOPLEVEL_PROJECT Autoscoper)
include(ExternalProject)
include(ExternalProjectDependency)

#-----------------------------------------------------------------------------
include(PreventInSourceBuilds)
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
if(NOT DEFINED Autoscoper_SAMPLE_DATA_DIR)
  set(Autoscoper_SAMPLE_DATA_DIR "${Autoscoper_BIN_DIR}")
endif()

#-----------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------
set(_default_backend "CUDA")
set(_force_backend )
if(DEFINED Autoscoper_BUILD_WITH_CUDA)
  set(_force_backend FORCE)
  if(NOT Autoscoper_BUILD_WITH_CUDA)
    set(_default_backend "OpenCL")
  endif()
  message(DEPRECATION "Consider setting Autoscoper_RENDERING_BACKEND to '${_default_backend}' instead of Autoscoper_BUILD_WITH_CUDA to '${Autoscoper_BUILD_WITH_CUDA}'")
endif()
set(Autoscoper_RENDERING_BACKEND "${_default_backend}" CACHE STRING "Backend to use for DRR and radiograph rendering" ${_force_backend})
set_property(CACHE Autoscoper_RENDERING_BACKEND PROPERTY STRINGS "CUDA" "OpenCL")
if(NOT Autoscoper_RENDERING_BACKEND MATCHES "^(CUDA|OpenCL)$")
  message(FATAL_ERROR "Autoscoper_RENDERING_BACKEND must be set to CUDA or OpenCL")
endif()
mark_as_superbuild(Autoscoper_RENDERING_BACKEND)
message(STATUS "Configuring with rendering backend '${Autoscoper_RENDERING_BACKEND}'")

option(Autoscoper_INSTALL_SAMPLE_DATA "Copy/Install the sample data to the build/install directory" ON)
mark_as_superbuild(Autoscoper_INSTALL_SAMPLE_DATA)

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
  mark_as_superbuild(VARS CMAKE_BUILD_TYPE ALL_PROJECTS)
else()
  mark_as_superbuild(VARS CMAKE_CONFIGURATION_TYPES ALL_PROJECTS)
endif()

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(BUILD_SHARED_LIBS ON)

if(Autoscoper_RENDERING_BACKEND STREQUAL "CUDA")
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
endif()

#-----------------------------------------------------------------------------
# Sample Data
#-----------------------------------------------------------------------------
set(Autoscoper_INSTALL_ROOT "./")
set(Autoscoper_BUNDLE_LOCATION "Autoscoper.app/Contents")
if(APPLE)
  set(Autoscoper_INSTALL_ROOT "${Autoscoper_BUNDLE_LOCATION}/") # Set to create Bundle
endif()

set(Autoscoper_INSTALL_SAMPLE_DATA_DIR "${Autoscoper_INSTALL_ROOT}${Autoscoper_SAMPLE_DATA_DIR}")

if(Autoscoper_INSTALL_SAMPLE_DATA)

  # Copy sample_data
  if(NOT CMAKE_CONFIGURATION_TYPES)
    set(_copy_subdir "")
  else()
    set(_copy_subdir "$<CONFIG>")
  endif()
  set(_copy_output "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/copy_autoscoper_sample_data_complete")
  add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${Autoscoper_SOURCE_DIR}/sample_data
      ${CMAKE_BINARY_DIR}/${Autoscoper_SAMPLE_DATA_DIR}/${_copy_subdir}/sample_data
    COMMAND ${CMAKE_COMMAND} -E touch ${_copy_output}
    OUTPUT ${_copy_output}
    COMMENT "Copying sample_data to build directory"
    )
  add_custom_target(CopyAutoscoperSampleData ALL
    DEPENDS
      ${_copy_output}
    )

  # Install sample_data
  install(DIRECTORY sample_data DESTINATION "${Autoscoper_INSTALL_SAMPLE_DATA_DIR}/$<CONFIG>" COMPONENT Runtime)
endif()
