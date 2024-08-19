set(libautoscoper_HEADERS
  src/asys/SystemTools.hxx
  src/Backtrace.hpp
  src/Camera.hpp
  src/CoordFrame.hpp
  src/DownhillSimplex.hpp
  src/Filter.hpp
  src/KeyCurve.hpp
  src/PSO.hpp
  src/TiffImage.h
  src/Tracker.hpp
  src/Trial.hpp
  src/Video.hpp
  src/View.hpp
  src/Volume.hpp
  src/VolumeDescription.hpp
  src/VolumeTransform.hpp
)

set(libautoscoper_SOURCES
  src/asys/SystemTools.cxx
  src/Backtrace.cpp
  src/Camera.cpp
  src/CoordFrame.cpp
  src/DownhillSimplex.cpp
  src/PSO.cpp
  src/TiffImage.cpp
  src/Tracker.cpp
  src/Trial.cpp
  src/Video.cpp
  src/View.cpp
  src/Volume.cpp
  src/VolumeDescription.cpp
  src/VolumeTransform.cpp
)

if(Autoscoper_BUILD_TESTING)
  list(APPEND libautoscoper_HEADERS
    src/vtkAddonTestingUtilities.h
  )
  list(APPEND libautoscoper_SOURCES
    src/vtkAddonTestingUtilities.cpp
    src/vtkAddonTestingUtilities.tpp
  )
endif()

find_package(Filesystem COMPONENTS Experimental Final)
if(CXX_FILESYSTEM_HEADER STREQUAL "experimental/filesystem")
  set(Autoscoper_HAS_EXPERIMENTAL_FILESYSTEM 1)
elseif(CXX_FILESYSTEM_HEADER STREQUAL "filesystem")
  set(Autoscoper_HAS_FILESYSTEM 1)
endif()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/filesystem_compat.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/filesystem_compat.hpp
  @ONLY
  )
list(APPEND libautoscoper_SOURCES
  src/filesystem_compat.cpp
  )

if(Autoscoper_RENDERING_BACKEND STREQUAL "CUDA")
  # CUDA 10.2 supports C++ up to version 14
  # See https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/index.html#c-cplusplus-language-support
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  if("$ENV{CUDAARCHS}" STREQUAL "" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
      set(_cuda_achitectures "all")
    else()
      # Adapted from https://github.com/Kitware/CMake/blob/v3.29.0-rc1/Modules/Internal/CMakeCUDAArchitecturesAll.cmake
      set(_cuda_achitectures
        50 52 53
        60 61 62 # >= 8.0
        70 72 # >= 9.0
        75 # >= 10.0
        80 # >= 11.0
        # 86 # >= 11.1
        # 87 # >= 11.4
        # 88 89 # >= 11.8
        )
      # only generate jit code for the newest arch
      list(POP_BACK _cuda_achitectures _latest_arch)
      list(TRANSFORM _cuda_achitectures APPEND "-real")
      list(APPEND _cuda_achitectures ${_latest_arch})
    endif()
    set(ENV{CUDAARCHS} "${_cuda_achitectures}")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  include(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda/CMakeLists.txt)
  add_library(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${cuda_HEADERS} ${cuda_SOURCES} ${cuda_KERNEL_HEADERS} ${cuda_KERNEL})
  target_include_directories(libautoscoper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda/cutil)
  target_link_libraries(libautoscoper PUBLIC
    CUDA::cudart_static
  )
  # The use of multiple .h and .cu files implies separate compilation units
  set_target_properties(libautoscoper PROPERTIES
    CMAKE_CUDA_SEPARABLE_COMPILATION ON
  )
elseif(Autoscoper_RENDERING_BACKEND STREQUAL "OpenCL")
  if(Autoscoper_OPENCL_USE_ICD_LOADER)
    find_package(OpenCLHeaders REQUIRED)
    find_package(OpenCLICDLoader REQUIRED)
  else()
    find_package(OpenCL ${Autoscoper_OpenCL_MINIMUM_REQUIRED_VERSION} REQUIRED)
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/opencl/CMakeLists.txt)
  add_library(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${opencl_SOURCES} ${opencl_HEADERS})
  add_dependencies(libautoscoper ${SHADER_TO_HEADER})
  target_compile_definitions(libautoscoper PUBLIC
    $<$<BOOL:${Autoscoper_OPENCL_USE_ICD_LOADER}>:CL_TARGET_OPENCL_VERSION=${Autoscoper_CL_TARGET_OPENCL_VERSION}>
    # Silence warning: 'glFunction' is deprecated: first deprecated in macOS 10.14 - OpenGL API deprecated
    $<$<AND:$<NOT:$<BOOL:${Autoscoper_OPENCL_USE_ICD_LOADER}>>,$<BOOL:${APPLE}>>:GL_SILENCE_DEPRECATION>
  )
  target_link_libraries(libautoscoper PUBLIC
    OpenCL::OpenCL
  )
else()
  message(FATAL_ERROR "Setting Autoscoper_RENDERING_BACKEND to '${Autoscoper_RENDERING_BACKEND}' is not supported")
endif()

target_compile_definitions(libautoscoper PUBLIC
  Autoscoper_RENDERING_USE_${Autoscoper_RENDERING_BACKEND}_BACKEND
)

set_target_properties(libautoscoper PROPERTIES
  RUNTIME_OUTPUT_NAME "libautoscoper${Autoscoper_ARTIFACT_SUFFIX}"
  LIBRARY_OUTPUT_NAME "libautoscoper${Autoscoper_ARTIFACT_SUFFIX}"
  ARCHIVE_OUTPUT_NAME "libautoscoper${Autoscoper_ARTIFACT_SUFFIX}"
)

set_target_properties(libautoscoper PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_LIB_DIR}"
  )

target_link_libraries(libautoscoper PUBLIC std::filesystem)

find_package(TIFF REQUIRED MODULE)
target_link_libraries(libautoscoper PUBLIC TIFF::TIFF)
target_compile_definitions(libautoscoper PUBLIC -DUSE_LIBTIFF)

find_package(JsonCpp REQUIRED)
target_link_libraries(libautoscoper PRIVATE ${JsonCpp_LIBRARY})
target_include_directories(libautoscoper PRIVATE ${JsonCpp_INCLUDE_DIR})

target_include_directories(libautoscoper PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ../math
)

install(TARGETS libautoscoper
  RUNTIME DESTINATION ${Autoscoper_BIN_DIR} COMPONENT Runtime
  LIBRARY DESTINATION ${Autoscoper_BIN_DIR} COMPONENT Runtime
  ARCHIVE DESTINATION ${Autoscoper_LIB_DIR} COMPONENT Development
)

if(Autoscoper_BUILD_TESTING)
  add_subdirectory(tests)
endif()
