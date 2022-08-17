set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

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

if(BUILD_WITH_CUDA)
	find_package( CUDA REQUIRED )
	set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
	include (${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda/CMakeLists.txt)
	#set(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
	CUDA_ADD_LIBRARY(libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${cuda_HEADERS} ${cuda_SOURCES} ${cuda_KERNEL_HEADERS} ${cuda_KERNEL})
	target_include_directories(libautoscoper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda/cutil)
	#add_dependencies(libautoscoper ${SHADER_TO_HEADER})


else()
	find_package( OpenCL REQUIRED )
	include (${CMAKE_CURRENT_SOURCE_DIR}/gpu/opencl/CMakeLists.txt)
	add_library (libautoscoper STATIC ${libautoscoper_SOURCES} ${libautoscoper_HEADERS} ${opencl_SOURCES} ${opencl_HEADERS})
	target_include_directories(libautoscoper PUBLIC  ${OpenCL_INCLUDE_DIRS} )
	add_dependencies(libautoscoper ${SHADER_TO_HEADER})
endif()

include(UseLIBTIFF)
UseLIBTIFF(libautoscoper PUBLIC)

target_include_directories(libautoscoper PUBLIC  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS libautoscoper DESTINATION lib)

