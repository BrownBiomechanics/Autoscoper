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
if(NOT DEFINED Qt5_DIR)
  message(FATAL_ERROR "Qt5_DIR is not defined")
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
set(Autoscoper_RENDERING_USE_${Autoscoper_RENDERING_BACKEND}_BACKEND 1)
message(STATUS "Configuring with rendering backend '${Autoscoper_RENDERING_BACKEND}'")

set(_default OFF)
if(WIN32)
  set(_default ON)
endif()
option(Autoscoper_INSTALL_Qt_LIBRARIES "Install Qt libraries" ${_default})
mark_as_superbuild(Autoscoper_INSTALL_Qt_LIBRARIES)

option(Autoscoper_INSTALL_DEPENDENCIES "Install dependencies" ON)
mark_as_superbuild(Autoscoper_INSTALL_DEPENDENCIES)

option(Autoscoper_INSTALL_SAMPLE_DATA "Copy/Install the sample data to the build/install directory" ON)
mark_as_superbuild(Autoscoper_INSTALL_SAMPLE_DATA)

option(Autoscoper_SUPERBUILD "Build ${PROJECT_NAME} and the projects it depends on." ON)
mark_as_advanced(Autoscoper_SUPERBUILD)
set(Autoscoper_BINARY_INNER_SUBDIR Autoscoper-build)

option(Autoscoper_BUILD_VTK "Builds VTK alongside Autoscoper." OFF)
mark_as_advanced(Autoscoper_BUILD_VTK)
mark_as_superbuild(Autoscoper_BUILD_VTK)

if(UNIX AND NOT APPLE)
  if(NOT DEFINED OpenGL_GL_PREFERENCE)
    set(OpenGL_GL_PREFERENCE "LEGACY")
  endif()
  if(NOT "${OpenGL_GL_PREFERENCE}" MATCHES "^(LEGACY|GLVND)$")
    message(FATAL_ERROR "OpenGL_GL_PREFERENCE variable is expected to be set to LEGACY or GLVND")
  endif()
  mark_as_superbuild(OpenGL_GL_PREFERENCE:STRING)
  message(STATUS "Setting OpenGL_GL_PREFERENCE to ${OpenGL_GL_PREFERENCE}")
endif()

if(Autoscoper_RENDERING_BACKEND STREQUAL "OpenCL")
  set(Autoscoper_OpenCL_MINIMUM_REQUIRED_VERSION "1.2.0")
  mark_as_superbuild(Autoscoper_OpenCL_MINIMUM_REQUIRED_VERSION:STRING)
  message(STATUS "Setting Autoscoper_OpenCL_MINIMUM_REQUIRED_VERSION to ${Autoscoper_OpenCL_MINIMUM_REQUIRED_VERSION}")

  set(Autoscoper_CL_TARGET_OPENCL_VERSION "120")
  mark_as_superbuild(Autoscoper_CL_TARGET_OPENCL_VERSION:STRING)
  message(STATUS "Setting Autoscoper_CL_TARGET_OPENCL_VERSION to ${Autoscoper_CL_TARGET_OPENCL_VERSION}")
endif()

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

#-----------------------------------------------------------------------------
# Collision Detection
#-----------------------------------------------------------------------------
if(Autoscoper_BUILD_VTK OR (DEFINED VTK_DIR AND EXISTS ${VTK_DIR}))
  message(STATUS "Configuring with collision detection")
  set(Autoscoper_COLLISION_DETECTION 1)
else()
  message(STATUS "Configuring without collision detection. If you want to enable collision detection, please set either Autoscoper_BUILD_VTK (if you want to build VTK as apart of the Superbuild) or VTK_DIR (If you want to use a pre-built version of VTK).")
  set(Autoscoper_COLLISION_DETECTION 0)
endif()
mark_as_superbuild(Autoscoper_COLLISION_DETECTION)

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

# Autoscoper_DEPENDENCIES and Autoscoper_SUPERBUILD_DIR CMake variables are set
# in SuperBuild.cmake.

#-----------------------------------------------------------------------------
# Launcher script
#-----------------------------------------------------------------------------
set(Autoscoper_CONFIGURE_LAUCHER_SCRIPT ON)
if(Autoscoper_CONFIGURE_LAUCHER_SCRIPT)
  if(WIN32)
    set(PATHVAR_CONFIG "Path")
    set(_pathsep ";")
    set(_input_script "autoscoper_set_env.bat.in")
    set(_launcher_script "autoscoper_set_env-$<CONFIG>.bat")
    set(_libdir "${Autoscoper_BIN_DIR}")
  elseif(APPLE)
    set(PATHVAR_CONFIG "DYLD_LIBRARY_PATH")
    set(_pathsep ":")
    set(_input_script "autoscoper_set_env.sh.in")
    set(_launcher_script "autoscoper_set_env.sh")
    set(_libdir "${Autoscoper_LIB_DIR}")
  else(UNIX)
    set(PATHVAR_CONFIG "LD_LIBRARY_PATH")
    set(_pathsep ":")
    set(_input_script "autoscoper_set_env.sh.in")
    set(_launcher_script "autoscoper_set_env.sh")
    set(_libdir "${Autoscoper_LIB_DIR}")
  endif()

  message(STATUS "Configuring '${_launcher_script}' with '${PATHVAR_CONFIG}' and '${_pathsep}'")

  set(_library_paths)

  # Qt
  get_property(_filepath TARGET "Qt5::Core" PROPERTY LOCATION_RELEASE)
  get_filename_component(_qt5Core_library_dir ${_filepath} PATH)
  list(APPEND _library_paths "${_qt5Core_library_dir}")

  # Dependencies
  foreach(dependency IN LISTS Autoscoper_DEPENDENCIES)
    set(_dep_dir "${Autoscoper_SUPERBUILD_DIR}/${dependency}-install")
    if(NOT EXISTS ${_dep_dir})
      set(_dep_dir "${Autoscoper_SUPERBUILD_DIR}/${dependency}-build")
    endif()
    if(NOT EXISTS ${_dep_dir})
      message(AUTHOR_WARNING "Unknown directory: ${_dep_dir}")
      continue()
    endif()
    if(_dep_dir MATCHES "-install$")
      list(APPEND _library_paths "${_dep_dir}/${_libdir}/")
    else()
      if(NOT CMAKE_CONFIGURATION_TYPES)
        list(APPEND _library_paths "${_dep_dir}/${_libdir}/")
      else()
        list(APPEND _library_paths "${_dep_dir}/${_libdir}/$<CONFIG>/")
      endif()
    endif()
  endforeach()
  if(Autoscoper_COLLISION_DETECTION AND NOT VTK IN_LIST Autoscoper_DEPENDENCIES) # If collision is enabled but VTK was not built as part of the superbuild
    list(APPEND _library_paths "${VTK_DIR}/bin/$<CONFIG>/")
  endif()

  string(REPLACE ";" "${_pathsep}" PATHS_CONFIG "${_library_paths}")

  file(READ CMake/${_input_script} contents)
  # Substitute @VAR@ with corresponding variable
  string(CONFIGURE "${contents}" contents @ONLY)
  # If it applies, substitute $<CONFIG> and generate one file per config
  set(file_permissions_option "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
    # Since FILE_PERMISSIONS option was introduced in CMake 3.20, we conditionally
    # specify it and use "add_custom_command()" in autoscoper/CMakeLists.txt
    set(file_permissions_option FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_WRITE GROUP_EXECUTE
      WORLD_READ
      )
  endif()
  file(GENERATE
    OUTPUT ${Autoscoper_BINARY_DIR}/${Autoscoper_BIN_DIR}/${_launcher_script}
    CONTENT "${contents}"
    ${file_permissions_option}
  )
endif()

#-----------------------------------------------------------------------------
# Install
#-----------------------------------------------------------------------------
if(Autoscoper_INSTALL_DEPENDENCIES)
  foreach(dependency IN LISTS Autoscoper_DEPENDENCIES)
    install(SCRIPT "${Autoscoper_SUPERBUILD_DIR}/${dependency}-build/cmake_install.cmake")
  endforeach()
  if(Autoscoper_COLLISION_DETECTION AND NOT VTK IN_LIST Autoscoper_DEPENDENCIES) # If collision is enabled but VTK was not built as part of the superbuild
    install(SCRIPT "${VTK_DIR}/cmake_install.cmake")
  endif()
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
  install(DIRECTORY sample_data DESTINATION "${Autoscoper_INSTALL_SAMPLE_DATA_DIR}" COMPONENT Runtime)
endif()
