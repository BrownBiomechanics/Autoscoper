set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(OpenGL)

#ADD QT4 SUPPORT
find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL Network REQUIRED)
cmake_policy(SET CMP0020 OLD)
set(CMAKE_AUTOMOC ON)

set(autoscoper_SOURCES
  src/main.cpp
)

#HEADER AND SOURCE FILES FOR UI
set(autoscoper_FORMS_HEADERS
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
  src/ui/AdvancedOptionsDialog.h
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
  src/ui/VolumeBox.h
  src/net/Socket.h
  src/ui/AboutAutoscoper.h
)
set(autoscoper_FORMS_SOURCES
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
  src/ui/AdvancedOptionsDialog.cpp
  src/ui/AboutAutoscoper.cpp
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
  src/ui/VolumeBox.cpp
  src/net/Socket.cpp
)

#QT DESIGNER UI FILES
set(autoscoper_FORMS
  src/ui/ui-files/AboutAutoscoper.ui
  src/ui/ui-files/AutoscoperMainWindow.ui
  src/ui/ui-files/FilterDockWidget.ui
  src/ui/ui-files/CameraViewWidget.ui
  src/ui/ui-files/TimelineDockWidget.ui
  src/ui/ui-files/ImportExportTrackingOptionsDialog.ui
  src/ui/ui-files/TrackingOptionsDialog.ui
  src/ui/ui-files/AdvancedOptionsDialog.ui
  src/ui/ui-files/OpenCLPlatformSelectDialog.ui
  src/ui/ui-files/CameraBox.ui
  src/ui/ui-files/NewTrialDialog.ui
  src/ui/ui-files/VolumeDockWidget.ui
  src/ui/ui-files/VolumeBox.ui
)

QT5_WRAP_UI(autoscoper_FORMS_HEADERS_GEN ${autoscoper_FORMS})
set(autoscoper_RESOURCES
  src/ui/autoscoper_resource.qrc
)
QT5_ADD_RESOURCES(autoscoper_RESOURCES_RCC ${autoscoper_RESOURCES})

# Set some Win32 Specific Settings

# Set some Apple MacOS Specific settings
if(APPLE)
  set(GUI_TYPE MACOSX_BUNDLE)
  #set(MACOSX_BUNDLE_ICON_FILE xromm.icns)
# Allows for bundle re-creation just by running "make". Also installs bundle icon
#add_custom_target(osx_bundle_dirs
#    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/Resources
#    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/MacOS
#    COMMAND cp ${CMAKE_SOURCE_DIR}/src/ui/ui-files/images/icons/${MACOSX_BUNDLE_ICON_FILE}
#            ${CMAKE_CURRENT_BINARY_DIR}/bin/Undistortion.app/Contents/Resources/${MACOSX_BUNDLE_ICON_FILE})
endif()

add_executable(autoscoper  ${GUI_TYPE} ${autoscoper_SOURCES} ${autoscoper_HEADERS}
  ${autoscoper_FORMS_SOURCES}
  ${autoscoper_FORMS_HEADERS}
  ${autoscoper_FORMS_HEADERS_GEN}
  ${autoscoper_RESOURCES_RCC}
  src/ui/autoscoper_resource.qrc
)

if(Autoscoper_BUILD_WITH_CUDA)
  find_package(CUDA REQUIRED)
  target_include_directories(autoscoper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cuda/cutil)
  set(GPU_LIBRARIES
    ${CUDA_LIBRARIES}
  )
else()
  find_package(OpenCL REQUIRED)
  target_include_directories(autoscoper PUBLIC ${OPENCL_INCLUDE_DIRS})
  set(GPU_LIBRARIES
    ${OpenCL_LIBRARIES}
  )
endif()

target_link_libraries(autoscoper PUBLIC libautoscoper
  ${OPENGL_LIBRARIES}
  ${GPU_LIBRARIES}
  ${Qt5Core_LIBRARIES}
  ${Qt5Widgets_LIBRARIES}
  ${Qt5OpenGL_LIBRARIES}
  ${Qt5Network_LIBRARIES}
  ${Qt5Gui_LIBRARIES})

target_include_directories(autoscoper PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(autoscoper PUBLIC
  ../math
  ../libautoscoper/src
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
  ${Qt5OpenGL_INCLUDE_DIRS}
  ${Qt5Network_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS})

find_package(GLEW)
target_link_libraries(autoscoper PUBLIC GLEW::GLEW)
target_compile_definitions(autoscoper PUBLIC -DUSE_GLEW)
find_package(TIFF)
target_link_libraries(autoscoper PUBLIC TIFF::TIFF)
target_compile_definitions(autoscoper PUBLIC -DUSE_LIBTIFF)

install(TARGETS autoscoper DESTINATION bin/Debug CONFIGURATIONS Debug)
install(TARGETS autoscoper DESTINATION bin/Release CONFIGURATIONS Release)

if(WIN32)
  install(FILES ${_qt5Core_install_prefix}/bin/Qt5OpenGL.dll ${_qt5Core_install_prefix}/bin/Qt5Widgets.dll ${_qt5Core_install_prefix}/bin/Qt5Gui.dll ${_qt5Core_install_prefix}/bin/Qt5Core.dll ${_qt5Core_install_prefix}/bin/Qt5Network.dll DESTINATION bin/Release CONFIGURATIONS Release)
  install(FILES ${_qt5Core_install_prefix}/bin/Qt5OpenGLd.dll ${_qt5Core_install_prefix}/bin/Qt5Widgetsd.dll ${_qt5Core_install_prefix}/bin/Qt5Guid.dll ${_qt5Core_install_prefix}/bin/Qt5Cored.dll ${_qt5Core_install_prefix}/bin/Qt5Networkd.dll DESTINATION bin/Debug CONFIGURATIONS Debug)
  install(FILES ${_qt5Core_install_prefix}/plugins/platforms/qwindows.dll DESTINATION bin/Release/platforms CONFIGURATIONS Release)
  install(FILES ${_qt5Core_install_prefix}/plugins/platforms/qwindowsd.dll DESTINATION bin/Debug/platforms CONFIGURATIONS Debug)
endif()

if(APPLE)
  file(COPY ../sample_data DESTINATION ../bin/Debug/autoscoper.app/Contents/MacOS)
endif()

