if(WIN32)
  install(
    FILES
      ${_qt5Core_install_prefix}/bin/Qt5OpenGL.dll
      ${_qt5Core_install_prefix}/bin/Qt5Widgets.dll
      ${_qt5Core_install_prefix}/bin/Qt5Gui.dll
      ${_qt5Core_install_prefix}/bin/Qt5Core.dll
      ${_qt5Core_install_prefix}/bin/Qt5Network.dll
    DESTINATION ${Autoscoper_BIN_DIR}/Release CONFIGURATIONS Release
    COMPONENT Runtime
  )
  install(
    FILES
      ${_qt5Core_install_prefix}/bin/Qt5OpenGLd.dll
      ${_qt5Core_install_prefix}/bin/Qt5Widgetsd.dll
      ${_qt5Core_install_prefix}/bin/Qt5Guid.dll
      ${_qt5Core_install_prefix}/bin/Qt5Cored.dll
      ${_qt5Core_install_prefix}/bin/Qt5Networkd.dll
    DESTINATION ${Autoscoper_BIN_DIR}/Debug CONFIGURATIONS Debug
    COMPONENT Runtime
  )
  install(
    FILES
      ${_qt5Core_install_prefix}/plugins/platforms/qwindows.dll
    DESTINATION ${Autoscoper_BIN_DIR}/Release/platforms CONFIGURATIONS Release
    COMPONENT Runtime
  )
  install(
    FILES
      ${_qt5Core_install_prefix}/plugins/platforms/qwindowsd.dll
    DESTINATION ${Autoscoper_BIN_DIR}/Debug/platforms CONFIGURATIONS Debug
    COMPONENT Runtime
  )
endif()
