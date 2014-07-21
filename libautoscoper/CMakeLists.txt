SET(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

find_package( TiffIO REQUIRED )

FIND_PACKAGE( OpenCV REQUIRED )

SET(libautoscoper_HEADERS 
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
)

SET(libautoscoper_SOURCES
    src/Backtrace.cpp
	src/Camera.cpp
	src/CoordFrame.cpp
	src/DownhillSimplex.cpp
	src/KeyCurve.cpp
	src/TiffImage.cpp
	src/Tracker.cpp
	src/Trial.cpp
	src/Video.cpp
	src/View.cpp
	src/Volume.cpp
	src/VolumeDescription.cpp
)

include_directories(${OPENCV_INCLUDE_DIRS} ${TIFFIO_INCLUDE_DIRS})

IF(BUILD_WITH_CUDA)
	find_package( CUDA REQUIRED )
	include (${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda/CMakeLists.txt)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda/cutil)
	#SET(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
	CUDA_ADD_LIBRARY(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${cuda_HEADERS} ${cuda_SOURCES} ${cuda_KERNEL_HEADERS} ${cuda_KERNEL})
ELSE(NOT BUILD_WITH_CUDA)
	find_package( OpenCL REQUIRED )
	include (${CMAKE_CURRENT_SOURCE_DIR}/gpu/opencl/CMakeLists.txt) 
	include_directories(${OPENCL_INCLUDE_DIRS} )
	add_library (libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${opencl_SOURCES} ${opencl_HEADERS})
	ADD_DEPENDENCIES(libautoscoper ${SHADER_TO_HEADER})
ENDIF()


