SET(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

FIND_PACKAGE(OpenGL)

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
		src/ui/VolumeDockWidget.h
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
		src/ui/VolumeDockWidget.cpp
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
		src/ui/ui-files/VolumeDockWidget.ui
)

QT4_WRAP_UI(autoscoper_FORMS_HEADERS_GEN ${autoscoper_FORMS})
SET(autoscoper_RESOURCES 
		src/ui/autoscoper_resource.qrc
)
QT4_ADD_RESOURCES(autoscoper_RESOURCES_RCC ${autoscoper_RESOURCES})


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
		
IF(BUILD_WITH_CUDA)
ELSE (NOT BUILD_WITH_CUDA)	
	find_package( OpenCL REQUIRED )
	target_include_directories(autoscoper PUBLIC ${OPENCL_INCLUDE_DIRS} )
	SET(GPU_LIBRARIES
		${OpenCL_LIBRARIES}
	)
ENDIF()

target_link_libraries(autoscoper PUBLIC libautoscoper ${OPENGL_LIBRARIES} ${GPU_LIBRARIES} ${QT_LIBRARIES})
target_include_directories(autoscoper PUBLIC  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(autoscoper PUBLIC  ../math ../libautoscoper/src) 

include(UseGLEW)
UseGLEW(autoscoper PUBLIC)

include(UseLIBTIFF) 
UseLIBTIFF(autoscoper PUBLIC)

install(TARGETS autoscoper DESTINATION bin/Debug CONFIGURATIONS Debug)
install(TARGETS autoscoper DESTINATION bin/Release CONFIGURATIONS Release)

get_filename_component(QT4_DIR ${QT_QMAKE_EXECUTABLE} DIRECTORY)
IF(WIN32)
	install(FILES ${QT4_DIR}/QtOpenGL4.dll ${QT4_DIR}/QtGui4.dll  ${QT4_DIR}/QtCore4.dll DESTINATION bin/Release CONFIGURATIONS Release)
	install(FILES ${QT4_DIR}/QtOpenGLd4.dll ${QT4_DIR}/QtGuid4.dll ${QT4_DIR}/QtCored4.dll DESTINATION bin/Debug CONFIGURATIONS Debug)
ENDIF(WIN32)
