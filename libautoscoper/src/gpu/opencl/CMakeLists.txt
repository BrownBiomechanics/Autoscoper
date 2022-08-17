include (src/gpu/opencl/kernel/CMakeLists.txt)

set(opencl_HEADERS
	src/gpu/opencl/Compositor.hpp
	src/gpu/opencl/ContrastFilter.hpp
	src/gpu/opencl/GaussianFilter.hpp
	src/gpu/opencl/Ncc.hpp
	src/gpu/opencl/OpenCL.hpp
	src/gpu/opencl/RadRenderer.hpp
	src/gpu/opencl/RayCaster.hpp
	src/gpu/opencl/SharpenFilter.hpp
	src/gpu/opencl/SobelFilter.hpp
	src/gpu/opencl/Merger.hpp
	src/gpu/opencl/BackgroundRenderer.hpp
	src/gpu/opencl/DRRBackground.hpp
	src/gpu/opencl/Mult.hpp
)

set(opencl_SOURCES
    src/gpu/opencl/Compositor.cpp
	src/gpu/opencl/ContrastFilter.cpp
	src/gpu/opencl/GaussianFilter.cpp
	src/gpu/opencl/Ncc.cpp
	src/gpu/opencl/OpenCL.cpp
	src/gpu/opencl/RadRenderer.cpp
	src/gpu/opencl/RayCaster.cpp
	src/gpu/opencl/SharpenFilter.cpp
	src/gpu/opencl/SobelFilter.cpp
	src/gpu/opencl/Merger.cpp
	src/gpu/opencl/BackgroundRenderer.cpp
	src/gpu/opencl/DRRBackground.cpp
	src/gpu/opencl/Mult.cpp
)


