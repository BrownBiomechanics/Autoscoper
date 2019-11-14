PROJECT(Autoscoper)

cmake_policy(SET CMP0048 NEW)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

option(BUILD_WITH_CUDA "Build With CUDA instead of OpenCL" ON)

#SET BUILD-TYPE TO RELEASE IF NOT DEFINED
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

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

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

IF(BUILD_WITH_CUDA)
	ADD_DEFINITIONS(-DWITH_CUDA)
ENDIF()

#ADD QT4 SUPPORT
FIND_PACKAGE(Qt5 COMPONENTS Core Widgets Gui REQUIRED)
#SET(QT_USE_QTOPENGL TRUE)
#INCLUDE(${QT_USE_FILE})
#SET(CMAKE_AUTOMOC ON)


set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(EXTERNAL_DIR_NAME external)
set(EXTERNAL_DIR ${PROJECT_ROOT}/${EXTERNAL_DIR_NAME})
file(MAKE_DIRECTORY ${EXTERNAL_DIR})


# When we autobuild dependencies, we install them in CMAKE_INSTALL_PREFIX, and then look for them
# there using find_package().  So, this makes sure the install prefix is in the path that 
# find_package() searches.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/external)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}/external)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/Modules/")

option(AUTOBUILD_DEPENDENCIES "If enabled, dependencies that cmake cannot find on the system will be downloaded to the `external' directory and built and installed to CMAKE_INSTALL_PREFIX during the configure process as a precursor to the build process." ON)

if (AUTOBUILD_DEPENDENCIES)
    message(STATUS "AUTOBUILD_DEPENDENCIES is ON.")
else()
    message(STATUS "AUTOBUILD_DEPENDENCIES is OFF.")
endif()

add_subdirectory (libautoscoper) 
add_subdirectory (autoscoper)

IF(WIN32)
	file(GLOB external_dlls "${CMAKE_INSTALL_PREFIX}/external/bin/*.dll")
	file(GLOB external_dllsd "${CMAKE_INSTALL_PREFIX}/external/bin/*d.dll")
	list(REMOVE_ITEM external_dlls ${external_dllsd})

	install(FILES ${external_dlls} DESTINATION bin/Release CONFIGURATIONS Release)
	install(FILES ${external_dllsd} DESTINATION bin/Debug CONFIGURATIONS Debug)
	
	file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/Release)

ENDIF(WIN32)

IF(UNIX)
	file(COPY sample_data DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
ENDIF(UNIX)
