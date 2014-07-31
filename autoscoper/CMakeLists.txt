SET(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(GLEW REQUIRED)

FIND_PACKAGE(GLUT REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

FIND_PACKAGE( OpenGL )

FIND_PACKAGE( OpenCV REQUIRED )

IF(NOT QT_TIFFIO)
	FIND_PACKAGE(TiffIO)
ENDIF()

#ADD QT4 SUPPORT
FIND_PACKAGE(Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL )
set(QT_USE_QTOPENGL TRUE)
INCLUDE(${QT_USE_FILE})
SET(CMAKE_AUTOMOC ON)
SET(autoscoper_SOURCES
     	src/main.cpp
)

#HEADER AND SOURCE FILES FOR UI
SET(autoscoper_FORMS_HEADERS 
		src/ui/AutoscoperMainWindow.h
		src/ui/FilterDockWidget.h
		src/ui/FilterTreeWidget.h
		src/ui/QtCategoryButton.h
		src/ui/CameraTreeWidgetItem.h
		src/ui/ModelViewTreeWidgetItem.h
		src/ui/FilterTreeWidgetItem.h
		src/ui/GLView.h
		src/ui/CameraViewWidget.h
		src/ui/TimelineDockWidget.h
		src/ui/ImportExportTrackingOptionsDialog.h
		src/ui/TrackingOptionsDialog.h
		src/ui/FilterTreeWidgetParameter.h
		src/ui/GLWidget.h
		src/ui/GLTimeline.h
		src/ui/GLTracker.h
		src/ui/Manip3D.hpp
		src/core/History.hpp
		src/ui/OpenCLPlatformSelectDialog.h
		src/ui/CameraBox.h
		src/ui/NewTrialDialog.h
		src/ui/WorldViewWindow.h
)
SET(autoscoper_FORMS_SOURCES
     	src/ui/AutoscoperMainWindow.cpp
		src/ui/FilterDockWidget.cpp
		src/ui/FilterTreeWidget.cpp
		src/ui/QtCategoryButton.cpp
		src/ui/CameraTreeWidgetItem.cpp
		src/ui/ModelViewTreeWidgetItem.cpp
		src/ui/FilterTreeWidgetItem.cpp
		src/ui/GLView.cpp
		src/ui/CameraViewWidget.cpp
		src/ui/TimelineDockWidget.cpp
		src/ui/ImportExportTrackingOptionsDialog.cpp
		src/ui/TrackingOptionsDialog.cpp
		src/ui/FilterTreeWidgetParameter.cpp
		src/ui/GLWidget.cpp
		src/ui/GLTimeline.cpp
		src/ui/GLTracker.cpp
		src/ui/Manip3D.cpp
		src/ui/OpenCLPlatformSelectDialog.cpp
		src/ui/CameraBox.cpp
		src/ui/NewTrialDialog.cpp
		src/ui/WorldViewWindow.cpp
)

#QT DESIGNER UI FILES
SET(autoscoper_FORMS 
		src/ui/ui-files/AutoscoperMainWindow.ui
		src/ui/ui-files/FilterDockWidget.ui
		src/ui/ui-files/CameraViewWidget.ui
		src/ui/ui-files/TimelineDockWidget.ui
		src/ui/ui-files/ImportExportTrackingOptionsDialog.ui
		src/ui/ui-files/TrackingOptionsDialog.ui
		src/ui/ui-files/OpenCLPlatformSelectDialog.ui
		src/ui/ui-files/CameraBox.ui
		src/ui/ui-files/NewTrialDialog.ui
)

QT4_WRAP_UI(autoscoper_FORMS_HEADERS_GEN ${autoscoper_FORMS})
SET(autoscoper_RESOURCES 
		src/ui/autoscoper_resource.qrc
)
QT4_ADD_RESOURCES(autoscoper_RESOURCES_RCC ${autoscoper_RESOURCES})

IF(BUILD_WITH_CUDA)
ELSE (NOT BUILD_WITH_CUDA)	
	find_package( OpenCL REQUIRED )
	include_directories(${OPENCL_INCLUDE_DIRS} )
	SET(GPU_LIBRARIES
		${OPENCL_LIBRARIES}
	)
ENDIF()

include_directories (${OPENCV_INCLUDE_DIRS} ${GLEW_INCLUDE_DIR} ${GLUT_INCLUDE_DIR}  ../math ../libautoscoper/src) 

link_directories (${LIBRARY_OUTPUT_PATH}) 


# Set some Win32 Specific Settings
IF(WIN32)
	SET(GUI_TYPE WIN32)
	SET(QT_USE_QMAIN true)
ENDIF(WIN32)

# Set some Apple MacOS Specific settings
IF (APPLE)
	SET(GUI_TYPE MACOSX_BUNDLE)
	#set( MACOSX_BUNDLE_ICON_FILE xromm.icns )
# Allows for bundle re-creation just by running "make". Also installs bundle icon
#add_custom_target(osx_bundle_dirs
#    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/Resources
#    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/MacOS
#    COMMAND cp ${CMAKE_SOURCE_DIR}/src/ui/ui-files/images/icons/${MACOSX_BUNDLE_ICON_FILE}
#            ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/Resources/${MACOSX_BUNDLE_ICON_FILE})
ENDIF (APPLE)

add_executable (autoscoper  ${GUI_TYPE} ${autoscoper_SOURCES} ${autoscoper_HEADERS}
		${autoscoper_FORMS_SOURCES} 
		${autoscoper_FORMS_HEADERS}
		${autoscoper_FORMS_HEADERS_GEN} 
		${autoscoper_RESOURCES_RCC} 
		src/ui/autoscoper_resource.qrc) 

IF(APPLE)
	TARGET_LINK_LIBRARIES(autoscoper libautoscoper ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${GPU_LIBRARIES} ${QT_LIBRARIES} ${TIFFIO_LIBRARIES} ${LZMA_LIBRARIES} )
	#Copy shared QT libs and build dog
	ADD_CUSTOM_COMMAND(TARGET autoscoper
			  POST_BUILD
			  COMMAND /opt/local/bin/macdeployqt ${CMAKE_BINARY_DIR}/bin/autoscoper.app/ -dmg
	)
ELSE()
	IF(QT_TIFFIO)
		TARGET_LINK_LIBRARIES(autoscoper libautoscoper ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${GPU_LIBRARIES} ${QT_LIBRARIES} ../Release/libtiff)
	ELSE()
		TARGET_LINK_LIBRARIES(autoscoper libautoscoper ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${GPU_LIBRARIES} ${QT_LIBRARIES} ${TIFFIO_LIBRARIES})
	ENDIF()
ENDIF()

