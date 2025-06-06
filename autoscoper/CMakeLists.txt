find_package(OpenGL)

find_package(Qt5 COMPONENTS Core Widgets Gui OpenGL Network REQUIRED)
set(CMAKE_AUTOMOC ON)

set(autoscoper_SOURCES
  src/main.cpp
)

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
  src/ui/TrackingSetDockWidget.h
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
  src/ui/VolumeListWidgetItem.h
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
  src/ui/TrackingSetDockWidget.cpp
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
  src/ui/VolumeListWidgetItem.cpp
  src/net/Socket.cpp
)

set(autoscoper_FORMS
  src/ui/ui-files/AboutAutoscoper.ui
  src/ui/ui-files/AutoscoperMainWindow.ui
  src/ui/ui-files/FilterDockWidget.ui
  src/ui/ui-files/CameraViewWidget.ui
  src/ui/ui-files/TimelineDockWidget.ui
  src/ui/ui-files/ImportExportTrackingOptionsDialog.ui
  src/ui/ui-files/TrackingOptionsDialog.ui
  src/ui/ui-files/TrackingSetDockWidget.ui
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

if(APPLE AND Autoscoper_MACOSX_BUNDLE)
  set(GUI_TYPE MACOSX_BUNDLE)
endif()

add_executable(autoscoper  ${GUI_TYPE} ${autoscoper_SOURCES} ${autoscoper_HEADERS}
  ${autoscoper_FORMS_SOURCES}
  ${autoscoper_FORMS_HEADERS}
  ${autoscoper_FORMS_HEADERS_GEN}
  ${autoscoper_RESOURCES_RCC}
  src/ui/autoscoper_resource.qrc
)

if(NOT "${Autoscoper_EXECUTABLE_LINK_FLAGS}" STREQUAL "")
  set_target_properties(autoscoper PROPERTIES LINK_FLAGS ${Autoscoper_EXECUTABLE_LINK_FLAGS})
endif()

if(Autoscoper_CONFIGURE_LAUCHER_SCRIPT AND CMAKE_VERSION VERSION_LESS "3.20" AND UNIX)
  set(_launcher_script "autoscoper_set_env.sh")
  add_custom_command(TARGET autoscoper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Changing '${_launcher_script}' permissions"
    COMMAND chmod 775
      ${Autoscoper_BINARY_DIR}/${Autoscoper_BIN_DIR}/${_launcher_script}
    )
endif()

set_target_properties(autoscoper PROPERTIES
  OUTPUT_NAME "autoscoper${Autoscoper_ARTIFACT_SUFFIX}"
)

set_target_properties(autoscoper PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_BIN_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Autoscoper_LIB_DIR}"
)

target_link_libraries(autoscoper PUBLIC libautoscoper
  ${OPENGL_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
  Qt5::Network
  Qt5::Gui
)

target_include_directories(autoscoper PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ../math
)

find_package(GLEW REQUIRED CONFIG)
target_link_libraries(autoscoper PUBLIC GLEW::GLEW)
target_compile_definitions(autoscoper PUBLIC -DUSE_GLEW)

target_compile_definitions(autoscoper PRIVATE $<$<BOOL:${Autoscoper_INSTALL_SAMPLE_DATA}>:Autoscoper_INSTALL_SAMPLE_DATA>)

install(TARGETS autoscoper DESTINATION ${Autoscoper_BIN_DIR} COMPONENT Runtime)

if(Autoscoper_INSTALL_Qt_LIBRARIES)
  include("${Autoscoper_SOURCE_DIR}/CMake/AutoscoperInstallQt.cmake")
endif()
