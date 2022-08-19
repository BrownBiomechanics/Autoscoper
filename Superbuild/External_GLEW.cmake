############
# This code was adapted from https://github.com/Slicer/Slicer/blob/main/SuperBuild/External_zlib.cmake
############
set(proj GLEW)

set(${proj}_DEPENDENCIES "")

if(Autoscoper_USE_SYSTEM_${proj})
	unset(GLEW_ROOT CACHE)
	find_package(GLEW REQUIRED)
	set(GLEW_INCLUDE_DIR ${LIBGLEW_INCLUDE_DIR})
	set(GLEW_LIBRARY ${LIBGLEW_LIBRARIES})
endif()

if(DEFINED GLEW_ROOT AND NOT EXISTS ${GLEW_ROOT})
	message(FATAL_ERROR "GLEW_ROOT variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED GLEW_ROOT AND NOT Autoscoper_USE_SYSTEM_${proj})
	
	set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
	set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
	set(EP_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/external)
	
	ExternalProject_Add(${proj}
		URL https://sourceforge.net/projects/glew/files/glew/2.2.0/glew-2.2.0.zip
		URL_MD5 970535b75b1b69ebd018a0fa05af63d1
		SOURCE_DIR ${EP_SOURCE_DIR}
		BINARY_DIR ${EP_BINARY_DIR}
		INSTALL_DIR ${EP_INSTALL_DIR}
		CMAKE_ARGS
			-S ${EP_SOURCE_DIR}/build/cmake/
			-B ${EP_BINARY_DIR}
		CMAKE_CACHE_ARGS
			-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
			-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
			-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
			-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
			-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
			-DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
			-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
		DEPENDS
		  ${${proj}_DEPENDENCIES}
    )
	set(GLEW_DIR ${EP_INSTALL_DIR})
	set(GLEW_ROOT ${GLEW_DIR})
	set(GLEW_INCLUDE_DIR ${GLEW_DIR}/include)
	if(WIN32)
		set(GLEW_LIBRARY ${GLEW_DIR}/lib/GLEW.lib)
	else()
		set(GLEW_LIBRARY ${GLEW_DIR}/lib/libGLEW.a)
	endif()
endif()