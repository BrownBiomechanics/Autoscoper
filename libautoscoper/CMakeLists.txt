set(libautoscoper_HEADERS
  src/Backtrace.hpp
  src/Camera.hpp
  src/CoordFrame.hpp
  src/DownhillSimplex.hpp
  src/Filter.hpp
  src/KeyCurve.hpp
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
  src/Backtrace.cpp
  src/Camera.cpp
  src/CoordFrame.cpp
  src/DownhillSimplex.cpp
  src/PSO_kernel.cpp
  src/KeyCurve.cpp
  src/TiffImage.cpp
  src/Tracker.cpp
  src/Trial.cpp
  src/Video.cpp
  src/View.cpp
  src/Volume.cpp
  src/VolumeDescription.cpp
  src/VolumeTransform.cpp
)

if(Autoscoper_RENDERING_BACKEND STREQUAL "CUDA")
  find_package(CUDA REQUIRED)
  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  include(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda/CMakeLists.txt)
  CUDA_ADD_LIBRARY(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${cuda_HEADERS} ${cuda_SOURCES} ${cuda_KERNEL_HEADERS} ${cuda_KERNEL})
  target_include_directories(libautoscoper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda/cutil)
elseif(Autoscoper_RENDERING_BACKEND STREQUAL "OpenCL")
  find_package(OpenCL REQUIRED)
  include(${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/opencl/CMakeLists.txt)
  add_library(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${opencl_SOURCES} ${opencl_HEADERS})
  target_include_directories(libautoscoper PUBLIC  ${OpenCL_INCLUDE_DIRS})
  add_dependencies(libautoscoper ${SHADER_TO_HEADER})
else()
  message(FATAL_ERROR "Setting Autoscoper_RENDERING_BACKEND to '${Autoscoper_RENDERING_BACKEND}' is not supported")
endif()

set_target_properties(libautoscoper PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_LIB_DIR}"
  )

find_package(TIFF REQUIRED MODULE)
target_link_libraries(libautoscoper PUBLIC TIFF::TIFF)
target_compile_definitions(libautoscoper PUBLIC -DUSE_LIBTIFF)

target_include_directories(libautoscoper PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

install(TARGETS libautoscoper
  RUNTIME DESTINATION ${Autoscoper_BIN_DIR} COMPONENT Runtime
  LIBRARY DESTINATION ${Autoscoper_BIN_DIR} COMPONENT Runtime
  ARCHIVE DESTINATION ${Autoscoper_LIB_DIR} COMPONENT Development
)

